import gc
import time
from typing import List, Tuple, Union

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, BatchEncoding, PreTrainedTokenizer

from src.utils import load_model_and_tokenizer

ce_loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
softmax_fn = torch.nn.Softmax(dim=-1)


class Detectors:

    def __init__(self, observer_model_hf_name: str, performer_model_hf_name: str, hugging_face_api_token: str, bits_and_bytes_quantization_config = None, use_binoculars=False):
        self.performer_model, self.performer_tokenizer = load_model_and_tokenizer(performer_model_hf_name, hugging_face_api_token, bits_and_bytes_quantization_config)

        if observer_model_hf_name == performer_model_hf_name:
            print(f"Observer and performer are the same model ({observer_model_hf_name}), reusing single model instance.")
            self.observer_model = self.performer_model
            self.observer_tokenizer = self.performer_tokenizer
        else:
            self.observer_model, self.observer_tokenizer = load_model_and_tokenizer(observer_model_hf_name, hugging_face_api_token, bits_and_bytes_quantization_config)

        self.performer_model.eval()
        self.observer_model.eval()


    def compute_telescope_perplexity(self, text: Union[str, List[str]], device: torch.device = "cuda:0"):
        performer_logits, observer_logits, encodings = self._compute_logits(text, self.performer_model, self.observer_model, self.performer_tokenizer, device=device)
        return self._compute_telescope_perplexity(encodings, performer_logits)


    def compute_all_metrics(self, text: Union[str, List[str]], device: torch.device = "cuda:0"):
        performer_logits, observer_logits, encodings = self._compute_logits(text, self.performer_model, self.observer_model, self.performer_tokenizer, device=device)
                
        metrics_dict = self._compute_metrics_from_logits(encodings, performer_logits, observer_logits)

        return metrics_dict



    @torch.inference_mode()
    def _compute_logits(
            self,
            text: str,
            performer_model: AutoModelForCausalLM,
            observer_model: AutoModelForCausalLM,
            tokenizer: PreTrainedTokenizer,
            device: torch.device = "cuda:0"
        ):
        """
        Produces the performer logits, observer logits and encodings for a given performer model
        """

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Cap the model's max length at a fallback limit
        safe_max_length = min(tokenizer.model_max_length, 10_000)

        text_encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=safe_max_length).to(device)
        
        performer_model_logits = performer_model(**text_encodings).logits
        observer_model_logits = observer_model(**text_encodings).logits

        return performer_model_logits, observer_model_logits, text_encodings




    def _compute_metrics_from_logits(
            self,
            text_encodings: BatchEncoding,
            performer_model_logits: torch.Tensor,
            observer_model_logits: torch.Tensor,
            device="cuda:0"
        ):
        """
        Computes various metrics from performer and observer model logits.
        Designed to be easy to add and remove for quick prototyping and experiments.

        Returns:
            dict[str, float]: Every metric that can be used to debug or evaluate the classifier.
        """

        text_input_ids: torch.Tensor = text_encodings["input_ids"]
        attention_mask = text_encodings["attention_mask"]

        performer_model_logits = performer_model_logits.to(torch.float32)
        observer_model_logits = observer_model_logits.to(torch.float32)

        performer_model_probabilities = torch.softmax(performer_model_logits, dim=-1, dtype=torch.float16)
        observer_model_probabilities = torch.softmax(observer_model_logits, dim=-1, dtype=torch.float16)

        # Calculate telescope perplexity (Telescope)
        telescope_perplexity = self._compute_telescope_perplexity(text_encodings, performer_model_logits)        

        # Normal Perplexity and Cross Perplexity (Binoculars)
        causal_perplexity = self._compute_perplexity(text_encodings, performer_model_logits)
        cross_perplexity = self._compute_entropy(observer_model_logits.to(device), performer_model_logits.to(device), text_encodings.to(device), self.performer_tokenizer.pad_token_id)

        # Calculate entropy
        performer_model_entropy = -torch.sum(performer_model_probabilities * torch.log2(performer_model_probabilities + 1e-10)) / text_input_ids.size(1)
        observer_model_entropy = -torch.sum(observer_model_probabilities * torch.log2(observer_model_probabilities + 1e-10)) / text_input_ids.size(1)

        # DetectLLM LRR
        log_rank_ratio = self._compute_log_rank_ratio(performer_model_logits, text_encodings)


        # Fast-DetectGPT 
        logits_ref_sliced = observer_model_logits[:, :-1]
        logits_score_sliced = observer_model_logits[:, :-1]
        labels_sliced = text_input_ids[:, 1:]

        fast_detectgpt = self._compute_fast_detectgpt(logits_ref_sliced, logits_score_sliced, labels_sliced)

        fast_detectgpt_val = fast_detectgpt.mean().item()



        # Log-Likelihood and Log-Rank
        log_likelihood = self._compute_log_likelihood(performer_model_logits, text_encodings)
        log_rank = self._compute_log_rank(performer_model_logits, text_encodings)

        # Per Token Metrics
        per_token_telescope_perplexity = self._compute_telescope_perplexity_per_token(text_encodings, performer_model_logits)
        per_token_perplexity = self._compute_perplexity_per_token(text_encodings, performer_model_logits)
        per_token_cross_perplexity = self._compute_cross_perplexity_per_token(text_encodings, performer_model_logits, observer_model_logits, self.performer_tokenizer.pad_token_id)

        # Calculate KL divergence
        kl_div = torch.sum(performer_model_probabilities * (torch.log2(performer_model_probabilities + 1e-10) - torch.log2(observer_model_probabilities + 1e-10))) / text_input_ids.size(1)

        # Entropy Ratio
        entropy_ratio = performer_model_entropy.item() / observer_model_entropy.item()

        # Distribution Shift
        performer_total_variation_distance = self._compute_total_variation_distance(performer_model_logits, text_input_ids, attention_mask)
        observer_total_variation_distance = self._compute_total_variation_distance(observer_model_logits, text_input_ids, attention_mask)

        # Distribution Overlap
        performer_distribution_overlap = self._compute_distribution_overlap(performer_model_logits, text_input_ids, attention_mask)
        observer_distribution_overlap = self._compute_distribution_overlap(observer_model_logits, text_input_ids, attention_mask)

        # Logits Standard Deviation
        performer_logits_std = self._compute_logits_std(performer_model_logits, attention_mask)
        observer_logits_std = self._compute_logits_std(observer_model_logits, attention_mask)



        return {
            "telescope_perplexity": float(telescope_perplexity),
            "binoculars_score": float(causal_perplexity)/ float(cross_perplexity),
            "perplexity": float(causal_perplexity),

            "lrr": log_rank_ratio,
            "fast_detectgpt": fast_detectgpt_val,

            "performer_model_total_variation_distance": performer_total_variation_distance,
            "observer_model_total_variation_distance": observer_total_variation_distance,

            "performer_model_distribution_overlap": performer_distribution_overlap,
            "observer_model_distribution_overlap": observer_distribution_overlap,

            "performer_model_logits_std": performer_logits_std,
            "observer_model_logits_std": observer_logits_std,

            "performer_model_entropy": performer_model_entropy.item(),
            "observer_model_entropy": observer_model_entropy.item(),
            "entropy_ratio": entropy_ratio,

            "kl_divergence": kl_div.item(),

            "log_likelihood": log_likelihood,
            "log_rank": log_rank,
            
            "cross_perplexity": float(cross_perplexity),
            "telescope_perplexity_divided_by_cross_perplexity": float(telescope_perplexity)/ float(cross_perplexity),

            "telescope_perplexity_per_token" : per_token_telescope_perplexity,
            "perplexity_per_token": per_token_perplexity,
            "cross_perplexity_per_token": per_token_cross_perplexity,
        }












    
    # --------------------------------------------------------------------------------------
    # Here are all of the implementations for the metrics tested in this paper (and some extra)!
    # These are all private to the class and are not meant to be directly used
    # --------------------------------------------------------------------------------------

    
    
    def _compute_telescope_perplexity(
            self,
            encoding: transformers.BatchEncoding,
            logits: torch.Tensor,
            median: bool = False,
            temperature: float = 1.0
        ):

        shifted_logits = logits[..., :-1, :].contiguous() / temperature
        shifted_labels = encoding.input_ids[..., :-1].contiguous()
        shifted_attention_mask = encoding.attention_mask[..., :-1].contiguous()

        if median:
            ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).masked_fill(~shifted_attention_mask.bool(), float("nan")))
            ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)

        else:
            ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) * shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
            ppl = ppl.to("cpu").float().numpy()

        return ppl


    def _compute_perplexity(
            self,
            encoding: transformers.BatchEncoding,
            logits: torch.Tensor,
            median: bool = False,
            temperature: float = 1.0
        ):
        # from the original Binoculars paper:
        # https://github.com/ahans30/Binoculars
        # https://arxiv.org/pdf/2401.12070
        # Copyright (c) 2023, Abhimanyu Hans, Avi Schwarzschild, Tom Goldstein


        shifted_logits = logits[..., :-1, :].contiguous() / temperature
        shifted_labels = encoding.input_ids[..., 1:].contiguous()
        shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

        if median:
            ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).masked_fill(~shifted_attention_mask.bool(), float("nan")))
            ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)

        else:
            ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) * shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
            ppl = ppl.to("cpu").float().numpy()

        return ppl


    def _compute_entropy(
            self,
            p_logits: torch.Tensor,
            q_logits: torch.Tensor,
            encoding: transformers.BatchEncoding,
            pad_token_id: int,
            median: bool = False,
            sample_p: bool = False,
            temperature: float = 1.0
        ):
        # from the original Binoculars paper:
        # https://github.com/ahans30/Binoculars
        # https://arxiv.org/pdf/2401.12070
        # Copyright (c) 2023, Abhimanyu Hans, Avi Schwarzschild, Tom Goldstein


        vocab_size = p_logits.shape[-1]
        total_tokens_available = q_logits.shape[-2]
        p_scores, q_scores = p_logits / temperature, q_logits / temperature

        p_proba = softmax_fn(p_scores).view(-1, vocab_size)

        if sample_p:
            p_proba = torch.multinomial(p_proba.view(-1, vocab_size), replacement=True, num_samples=1).view(-1)

        q_scores = q_scores.view(-1, vocab_size)

        ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
        padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

        if median:
            ce_nan = ce.masked_fill(~padding_mask.bool(), float("nan"))
            agg_ce = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
        else:
            agg_ce = (((ce * padding_mask).sum(1) / padding_mask.sum(1)).to("cpu").float().numpy())


        return agg_ce


    def _compute_log_rank_ratio(self, logits, text_encodings) -> float:
        # This technique comes from the DetectLLM paper whose attribution can be found below
        # https://github.com/mbzuai-nlp/DetectLLM
        # https://arxiv.org/pdf/2306.05540

        log_probs, ranks = self._compute_token_probabilities_and_ranks(logits, text_encodings)
        log_ranks = [np.log(rank) for rank in ranks]

        sum_log_probs = sum(log_probs)
        sum_log_ranks = sum(log_ranks)

        # Take absolute ratio as defined in the paper
        return abs(sum_log_probs / (sum_log_ranks+1e-6))
    


    def _compute_fast_detectgpt(self, logits_ref: torch.Tensor, logits_score: torch.Tensor, labels: torch.Tensor):
        """
        Computes the Fast-DetectGPT metric: fast_detectgpt (analytic discrepancy).

        NOTE: This is only the analytic version of Fast-DetectGPT.
        For the sampling-based non-analytic version, see _compute_fast_detectgpt_non_analytic_debug.
        """
        # Ensure shapes match
        if logits_ref.size(-1) != logits_score.size(-1):
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        lprobs_score = torch.log_softmax(logits_score, dim=-1)
        probs_ref = torch.softmax(logits_ref, dim=-1)

        labels_expanded = labels.unsqueeze(-1)
        log_likelihood = lprobs_score.gather(dim=-1, index=labels_expanded).squeeze(-1)

        mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
        var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)

        fast_detectgpt = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()

        return fast_detectgpt







    def _compute_total_variation_distance(self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute total variation distance between observer's predictions and actual next tokens.

        Args:
            logits: Shape (batch_size, sequence_length, vocab_size)
            input_ids: Shape (batch_size, sequence_length)
            attention_mask: Shape (batch_size, sequence_length)

        Returns:
            Tensor of shape (batch_size,) containing TV distance for each sequence
        """

        probs = torch.softmax(logits, dim=-1)  # (batch_size, sequence_length, vocab_size)

        next_tokens = input_ids[:, 1:]  # (batch_size, sequence_length-1)
        probs = probs[:, :-1]  # Remove last prediction since we don't have the next token

        vocab_size = probs.size(-1)
        actual_probs = torch.zeros_like(probs)
        batch_indices = torch.arange(next_tokens.size(0)).unsqueeze(1).expand(-1, next_tokens.size(1))
        seq_indices = torch.arange(next_tokens.size(1)).unsqueeze(0).expand(next_tokens.size(0), -1)
        actual_probs[batch_indices, seq_indices, next_tokens] = 1

        abs_diff = torch.abs(probs - actual_probs)

        token_tv = 0.5 * torch.sum(abs_diff, dim=-1)  # (batch_size, sequence_length-1)

        mask = attention_mask[:, 1:]
        seq_lengths = mask.sum(dim=1).float()        # (batch_size,)
        masked_tv = token_tv * mask
        avg_tv = masked_tv.sum(dim=1) / seq_lengths  # (batch_size,)

        return avg_tv


    def _compute_distribution_overlap(self, logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute overlap across all token position distributions simultaneously.

        Args:
            logits: Shape (batch_size, sequence_length, vocab_size)
            input_ids: Shape (batch_size, sequence_length)
            attention_mask: Shape (batch_size, sequence_length)

        Returns:
            Tensor of shape (batch_size,) containing overlap score for each sequence
        """
        probs = torch.softmax(logits, dim=-1)

        next_tokens = input_ids[:, 1:]
        probs = probs[:, :-1]

        vocab_size = probs.size(-1)
        actual_probs = torch.zeros_like(probs)
        batch_indices = torch.arange(next_tokens.size(0)).unsqueeze(1).expand(-1, next_tokens.size(1))
        seq_indices = torch.arange(next_tokens.size(1)).unsqueeze(0).expand(next_tokens.size(0), -1)
        actual_probs[batch_indices, seq_indices, next_tokens] = 1

        mask = attention_mask[:, 1:].unsqueeze(-1)
        probs = probs * mask
        actual_probs = actual_probs * mask

        overlap = torch.minimum(probs, actual_probs)
        position_overlap = torch.sum(overlap, dim=-1)
        seq_lengths = mask.squeeze(-1).sum(dim=1).float()
        total_overlap = position_overlap.sum(dim=1) / seq_lengths

        return total_overlap



    def _compute_logits_std(self, logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)

        masked_probs = probs * attention_mask.unsqueeze(-1)

        batch_size = probs.size(0)
        flattened = masked_probs.view(batch_size, -1)
        std = torch.std(flattened, dim=1)
        return std



    def _compute_token_probabilities_and_ranks(self, logits, text_encodings) -> Tuple[List[float], List[int]]:
        probs = torch.softmax(logits, dim=-1)

        target_ids = text_encodings.input_ids[:, 1:-1]

        log_probs = []
        ranks = []

        for i in range(target_ids.shape[1]):
            token_id = target_ids[0, i].item()
            token_probs = probs[0, i]

            token_prob = token_probs[token_id].item()
            log_probs.append(np.log(token_prob) if token_prob > 0 else -float('inf'))

            sorted_probs, sorted_indices = torch.sort(token_probs, descending=True)
            rank = (sorted_indices == token_id).nonzero().item() + 1
            ranks.append(rank)

        return log_probs, ranks


    def _compute_log_likelihood(self, logits, text_encodings) -> float:
        log_probs, _ = self._compute_token_probabilities_and_ranks(logits, text_encodings)
        return sum(log_probs) / (len(log_probs)+1e-6)


    def _compute_log_rank(self, logits, text_encodings) -> float:
        _, ranks = self._compute_token_probabilities_and_ranks(logits, text_encodings)
        log_ranks = [np.log(rank) for rank in ranks]
        return sum(log_ranks) / (len(log_ranks) + 1e-6)









    def _compute_telescope_perplexity_per_token(
            self,
            encoding: transformers.BatchEncoding,
            logits: torch.Tensor,
            temperature: float = 1.0
        ):
        shifted_logits = logits[..., :-1, :].contiguous() / temperature
        shifted_labels = encoding.input_ids[..., :-1].contiguous()
        shifted_attention_mask = encoding.attention_mask[..., :-1].contiguous()

        token_loss = ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels)

        masked_token_loss = token_loss * shifted_attention_mask

        # Convert to numpy and split into per-sample lists
        # Only include losses where attention mask is 1
        token_loss_np = masked_token_loss.cpu().float().numpy()  # shape: (batch, seq_len)
        per_sample_losses = [
            sample_loss[sample_mask > 0].tolist()
            for sample_loss, sample_mask in zip(token_loss_np, shifted_attention_mask.cpu().numpy())
        ]

        return per_sample_losses



    def _compute_perplexity_per_token(
            self,
            encoding: transformers.BatchEncoding,
            logits: torch.Tensor,
            temperature: float = 1.0
        ):
        shifted_logits = logits[..., :-1, :].contiguous() / temperature
        shifted_labels = encoding.input_ids[..., 1:].contiguous()
        shifted_attention_mask = encoding.attention_mask[..., 1:].contiguous()

        token_loss = ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels)

        masked_token_loss = token_loss * shifted_attention_mask

        # Convert to numpy and split into per-sample lists
        # Only include losses where attention mask is 1
        token_loss_np = masked_token_loss.cpu().float().numpy()  # shape: (batch, seq_len)
        per_sample_losses = [
            sample_loss[sample_mask > 0].tolist()
            for sample_loss, sample_mask in zip(token_loss_np, shifted_attention_mask.cpu().numpy())
        ]

        return per_sample_losses


    def _compute_cross_perplexity_per_token(
            self,
            encoding: transformers.BatchEncoding,
            performer_logits: torch.Tensor,
            observer_logits: torch.Tensor,
            pad_token_id: int,
            temperature: float = 1.0
        ):

        vocab_size = observer_logits.shape[-1]
        total_tokens_available = performer_logits.shape[-2]
        p_scores, q_scores = observer_logits / temperature, performer_logits / temperature

        p_proba = softmax_fn(p_scores).view(-1, vocab_size)

        q_scores = q_scores.view(-1, vocab_size)

        ce = ce_loss_fn(input=q_scores, target=p_proba).view(-1, total_tokens_available)
        padding_mask = (encoding.input_ids != pad_token_id).type(torch.uint8)

        ce_per_token = (ce * padding_mask).to("cpu").float().numpy().tolist()

        return ce_per_token



    def _compute_fast_detectgpt_non_analytic_debug(self, logits_ref: torch.Tensor, logits_score: torch.Tensor, labels: torch.Tensor):
        """
        DEBUG ONLY: Computes the non-analytic (sample-based) version of Fast-DetectGPT.
        This is not used in the production calculation flow.
        """
        # Ensure shapes match
        if logits_ref.size(-1) != logits_score.size(-1):
            vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :, :vocab_size]
            logits_score = logits_score[:, :, :vocab_size]

        lprobs_score = torch.log_softmax(logits_score, dim=-1)
        labels_expanded = labels.unsqueeze(-1)
        log_likelihood = lprobs_score.gather(dim=-1, index=labels_expanded).squeeze(-1)

        nsamples = 10000
        lprobs_ref = torch.log_softmax(logits_ref, dim=-1)
        distrib = torch.distributions.categorical.Categorical(logits=lprobs_ref)
        samples = distrib.sample([nsamples]).permute(1, 2, 0) # [batch, seq, nsamples]

        log_likelihood_x_samples = lprobs_score.gather(dim=-1, index=samples) # [batch, seq, nsamples]
        log_likelihood_x_tilde = log_likelihood_x_samples.mean(dim=1) # Mean over sequence length -> [batch, nsamples]

        miu_tilde = log_likelihood_x_tilde.mean(dim=-1) # Mean over nsamples -> [batch]
        sigma_tilde = log_likelihood_x_tilde.std(dim=-1) # Std over nsamples -> [batch]

        log_likelihood_x = log_likelihood.mean(dim=1) # [batch]

        discrepancy = (log_likelihood_x - miu_tilde) / sigma_tilde

        return discrepancy





    def _compute_telescope_perplexity_and_cross_perplexity_from_logits(
            self,
            text_encodings: BatchEncoding,
            performer_logits: torch.Tensor,
            observer_logits: torch.Tensor,
            reference_offset: int = 0,
            number_of_tokens_to_skip: int = 20,
            device="cuda:0"
        ):
        """
        This is an alternative implementation of the telescope formula designed for debugging/ablation purposes.
        It is designed to be easier to understand and easier to reason through.

        Args:
            text_encodings (BatchEncoding): The BatchEncodings object of the text. Can be obtained from _compute_logits()
            performer_logits (torch.Tensor): The logits output from the performer model as a tensor. Can be obtained from _compute_logits()
            observer_logits (torch.Tensor): The logits output from the observer model as a tensor. Can be obtained from _compute_logits()
            device (str, optional): The device (GPU, CPU, or NPU) to perform calculations on. Defaults to "cuda:0".
            reference_offset (int, optional): This represents an offset to which token you use as a label when calculating the perplexity.
                Defaults to 0 and should only be changed for debugging/ablation purposes.
            number_of_tokens_to_skip (int, optional): When averaging the score across all tokens, skip the first 20 tokens.
                This allows the model to understand the context somewhat before we useits output to compute
                the scores. Changing this may slightly improve performance. Defaults to 20 tokens.

        Returns:
            tuple: The telescope perplexity and cross perplexity of the text.
        """

        observer_logits = observer_logits.squeeze()
        performer_logits = performer_logits.squeeze()
        
        text_input_ids: torch.Tensor = text_encodings["input_ids"]
        text_input_ids = text_input_ids.squeeze()

        total_cross_entropy_cross_perplexity = 0
        total_cross_entropy_normal_perplexity = 0
        tokens_evaluated = 0

        for current_token_index in range(performer_logits.shape[0] - reference_offset):
            if current_token_index < number_of_tokens_to_skip: continue

            tokens_evaluated += 1

            performer_next_token_logits = performer_logits[current_token_index, :].reshape(1, -1)
            observer_next_token_logits = observer_logits[current_token_index, :].reshape(1, -1)

            performer_next_tokens_logits_softmax = torch.softmax(performer_next_token_logits, dim=-1)
            observer_next_token_logits_softmax = torch.softmax(observer_next_token_logits, dim=-1)


            total_cross_entropy_cross_perplexity -= torch.matmul(performer_next_tokens_logits_softmax, torch.log(observer_next_token_logits_softmax).T)
            total_cross_entropy_normal_perplexity -= torch.log(performer_next_tokens_logits_softmax[:, text_input_ids[current_token_index + reference_offset]])

            del observer_next_token_logits_softmax
            del performer_next_tokens_logits_softmax
            del performer_next_token_logits
            del observer_next_token_logits
            torch.cuda.empty_cache()

        if tokens_evaluated > 0:
            normal_ppl = float(total_cross_entropy_normal_perplexity[0].cpu()) / tokens_evaluated
            cross_ppl = float(total_cross_entropy_cross_perplexity[0][0].cpu()) / tokens_evaluated
        else:
            normal_ppl = 0.0
            cross_ppl = 0.0

        return normal_ppl, cross_ppl