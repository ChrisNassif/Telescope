import gc
import time
from typing import List, Tuple, Union, Dict, Any, Optional

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, BatchEncoding, PreTrainedTokenizer, PreTrainedModel

from .utils import load_model_and_tokenizer, get_hugging_face_auth_token

cross_entropy_loss_function: torch.nn.CrossEntropyLoss = torch.nn.CrossEntropyLoss(reduction="none")
softmax_function: torch.nn.Softmax = torch.nn.Softmax(dim=-1)


class Detectors:
    performer_model: PreTrainedModel
    performer_tokenizer: PreTrainedTokenizer
    observer_model: PreTrainedModel
    observer_tokenizer: PreTrainedTokenizer
    device: Union[str, torch.device]

    def __init__(
            self, 
            performer_model_huggingface_name: str, 
            observer_model_huggingface_name: Optional[str] = None, 
            bits_and_bytes_quantization_config: Optional[Any] = None, 
            device: Optional[Union[str, torch.device]] = None
        ) -> None:

        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        hugging_face_auth_token: str = get_hugging_face_auth_token()
        
        self.performer_model, self.performer_tokenizer = load_model_and_tokenizer(
            performer_model_huggingface_name, hugging_face_auth_token, bits_and_bytes_quantization_config, device
        )

        if observer_model_huggingface_name == performer_model_huggingface_name or observer_model_huggingface_name is None:
            print(f"INFO: Observer and performer are the same model ({observer_model_huggingface_name}), reusing single model instance.")
            self.observer_model = self.performer_model
            self.observer_tokenizer = self.performer_tokenizer
        else:
            self.observer_model, self.observer_tokenizer = load_model_and_tokenizer(
                observer_model_huggingface_name, hugging_face_auth_token, bits_and_bytes_quantization_config, device
            )

        self.device = device
        self.performer_model.eval()
        self.observer_model.eval()


    def compute_telescope_perplexity(self, text: Union[str, List[str]], device: Optional[Union[str, torch.device]] = None) -> float:
        """
        Returns:
            float: The telescope perplexity of the given text.
        """
        if device is None:
            device = self.device
        performer_model_logits: torch.Tensor
        observer_model_logits: torch.Tensor
        text_encodings: BatchEncoding
        performer_model_logits, observer_model_logits, text_encodings = self._compute_logits(text, self.performer_model, self.observer_model, self.performer_tokenizer, device=device)
        performer_model_logits = performer_model_logits.to(torch.float32)
        return self._compute_telescope_perplexity(text_encodings, performer_model_logits)

    def compute_perplexity(self, text: Union[str, List[str]], device: Optional[Union[str, torch.device]] = None) -> float:
        """
        Returns:
            float: The perplexity of the given text.
        """
        if device is None:
            device = self.device
        performer_model_logits: torch.Tensor
        observer_model_logits: torch.Tensor
        text_encodings: BatchEncoding
        performer_model_logits, observer_model_logits, text_encodings = self._compute_logits(text, self.performer_model, self.observer_model, self.performer_tokenizer, device=device)
        performer_model_logits = performer_model_logits.to(torch.float32)        
        return self._compute_perplexity(text_encodings, performer_model_logits)


    def compute_binoculars_score(self, text: Union[str, List[str]], device: Optional[Union[str, torch.device]] = None) -> float:
        """
        Returns:
            float: The binoculars score of the given text.
        """
        if device is None:
            device = self.device
        performer_model_logits: torch.Tensor
        observer_model_logits: torch.Tensor
        text_encodings: BatchEncoding
        performer_model_logits, observer_model_logits, text_encodings = self._compute_logits(text, self.performer_model, self.observer_model, self.performer_tokenizer, device=device)
        
        performer_model_logits = performer_model_logits.to(torch.float32)
        observer_model_logits = observer_model_logits.to(torch.float32)

        causal_perplexity: float = self._compute_perplexity(text_encodings, performer_model_logits)
        cross_perplexity: float = self._compute_cross_perplexity(text_encodings, observer_model_logits, performer_model_logits, self.performer_tokenizer.pad_token_id)

        return float(causal_perplexity) / float(cross_perplexity)

    def compute_detectllm_log_rank_ratio(self, text: Union[str, List[str]], device: Optional[Union[str, torch.device]] = None) -> float:
        """
        Returns:
            float: The log rank ratio of the given text.
        """
        if device is None:
            device = self.device
        performer_model_logits: torch.Tensor
        observer_model_logits: torch.Tensor
        text_encodings: BatchEncoding
        performer_model_logits, observer_model_logits, text_encodings = self._compute_logits(text, self.performer_model, self.observer_model, self.performer_tokenizer, device=device)
        performer_model_logits = performer_model_logits.to(torch.float32)
        return self._compute_log_rank_ratio(text_encodings, performer_model_logits)


    def compute_fast_detectgpt(self, text: Union[str, List[str]], device: Optional[Union[str, torch.device]] = None) -> float:
        """
        Returns:
            float: The Fast DetectGPT score of the given text.
        """
        if device is None:
            device = self.device
        performer_model_logits: torch.Tensor
        observer_model_logits: torch.Tensor
        text_encodings: BatchEncoding
        performer_model_logits, observer_model_logits, text_encodings = self._compute_logits(text, self.performer_model, self.observer_model, self.performer_tokenizer, device=device)
        observer_model_logits = observer_model_logits.to(torch.float32)

        return self._compute_fast_detectgpt(text_encodings, observer_model_logits, observer_model_logits)
        


    def compute_all_metrics(self, text: Union[str, List[str]], device: Optional[Union[str, torch.device]] = None) -> Dict[str, Any]:
        """
        Computes various metrics from performer and observer model logits.

        Returns:
            dict[str, float]: Every metric that can be used to debug or evaluate the classifier. 
                results["metric_name"] = metric_value
        """
        if device is None:
            device = self.device
        performer_model_logits: torch.Tensor
        observer_model_logits: torch.Tensor
        text_encodings: BatchEncoding
        performer_model_logits, observer_model_logits, text_encodings = self._compute_logits(text, self.performer_model, self.observer_model, self.performer_tokenizer, device=device)
        metrics_dict: Dict[str, Any] = self._compute_metrics_from_logits(text_encodings, performer_model_logits, observer_model_logits, device=device)
        return metrics_dict



    @torch.inference_mode()
    def _compute_logits(
            self,
            text: Union[str, List[str]],
            performer_model: PreTrainedModel,
            observer_model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            device: Union[str, torch.device] = "cuda:0"
        ) -> Tuple[torch.Tensor, torch.Tensor, BatchEncoding]:
        """
        Produces the performer logits, observer logits and text encodings for a given performer model
        """

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Cap the model's max length at a fallback limit
        safe_max_length: int = min(tokenizer.model_max_length, 10_000)

        text_encodings: BatchEncoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=safe_max_length).to(device)
        
        performer_model_logits: torch.Tensor = performer_model(**text_encodings).logits.squeeze(0)

        observer_model_logits: torch.Tensor
        if performer_model != observer_model:
            observer_model_logits = observer_model(**text_encodings).logits.squeeze(0)
        else:
            observer_model_logits = performer_model_logits

        key: str
        for key in list(text_encodings.keys()):
            text_encodings[key] = text_encodings[key].squeeze(0)

        return performer_model_logits, observer_model_logits, text_encodings




    def _compute_metrics_from_logits(
            self,
            text_encodings: BatchEncoding,
            performer_model_logits: torch.Tensor,
            observer_model_logits: torch.Tensor,
            device: Union[str, torch.device] = "cuda:0"
        ) -> Dict[str, Any]:
        """
        Computes various metrics from performer and observer model logits.
        We designed this to be easy to add and remove different metrics 
        for quick prototyping and experiments.

        Returns:
            dict[str, float]: Every metric that can be used to debug or evaluate the classifier.
        """

        text_input_ids: torch.Tensor = text_encodings["input_ids"]
        attention_mask: torch.Tensor = text_encodings["attention_mask"]

        performer_model_logits = performer_model_logits.to(torch.float32)
        observer_model_logits = observer_model_logits.to(torch.float32)


        # Logits are converted to probabilities via softmax
        performer_model_probabilities: torch.Tensor = torch.softmax(performer_model_logits, dim=-1, dtype=torch.float32)
        observer_model_probabilities: torch.Tensor = torch.softmax(observer_model_logits, dim=-1, dtype=torch.float32)


        # Calculate telescope perplexity (Telescope)
        telescope_perplexity: float = self._compute_telescope_perplexity(text_encodings, performer_model_logits)        

        # Normal Perplexity and Cross Perplexity (Binoculars)
        causal_perplexity: float = self._compute_perplexity(text_encodings, performer_model_logits)
        cross_perplexity: float = self._compute_cross_perplexity(text_encodings, observer_model_logits, performer_model_logits, self.performer_tokenizer.pad_token_id)

        # DetectLLM Log Rank Ratio
        log_rank_ratio: float = self._compute_log_rank_ratio(text_encodings, performer_model_logits)

        # Fast-DetectGPT 
        fast_detectgpt_score: float = self._compute_fast_detectgpt(text_encodings, observer_model_logits, observer_model_logits)



        # Calculate entropy
        performer_model_entropy: torch.Tensor = -torch.sum(performer_model_probabilities * torch.log2(performer_model_probabilities + 1e-10)) / text_input_ids.size(0)
        observer_model_entropy: torch.Tensor = -torch.sum(observer_model_probabilities * torch.log2(observer_model_probabilities + 1e-10)) / text_input_ids.size(0)

        # Log-Likelihood and Log-Rank
        log_likelihood: float = self._compute_log_likelihood(text_encodings, performer_model_logits)
        log_rank: float = self._compute_log_rank(text_encodings, performer_model_logits)

        # Per Token Metrics
        per_token_telescope_perplexity: List[float] = self._compute_telescope_perplexity_per_token(text_encodings, performer_model_logits)
        per_token_perplexity: List[float] = self._compute_perplexity_per_token(text_encodings, performer_model_logits)
        per_token_cross_perplexity: List[float] = self._compute_cross_perplexity_per_token(text_encodings, performer_model_logits, observer_model_logits, self.performer_tokenizer.pad_token_id)

        # Calculate KL divergence
        kl_div: torch.Tensor = torch.sum(performer_model_probabilities * (torch.log2(performer_model_probabilities + 1e-10) - torch.log2(observer_model_probabilities + 1e-10))) / text_input_ids.size(0)

        # Entropy Ratio
        entropy_ratio: float = performer_model_entropy.item() / (observer_model_entropy.item() + 1e-10)

        # Distribution Shift
        performer_total_variation_distance: float = self._compute_total_variation_distance(text_encodings, performer_model_logits)
        observer_total_variation_distance: float = self._compute_total_variation_distance(text_encodings, observer_model_logits)

        # Distribution Overlap
        performer_distribution_overlap: float = self._compute_distribution_overlap(text_encodings, performer_model_logits)
        observer_distribution_overlap: float = self._compute_distribution_overlap(text_encodings, observer_model_logits)

        # Logits Standard Deviation
        performer_model_logits_standard_deviation: float = self._compute_logits_standard_deviation(text_encodings, performer_model_logits)
        observer_model_logits_standard_deviation: float = self._compute_logits_standard_deviation(text_encodings, observer_model_logits)



        return {
            "telescope_perplexity": float(telescope_perplexity),
            "binoculars_score": float(causal_perplexity)/ float(cross_perplexity),
            "perplexity": float(causal_perplexity),

            "log_rank_ratio": log_rank_ratio,
            "fast_detectgpt": fast_detectgpt_score,

            "performer_model_total_variation_distance": performer_total_variation_distance,
            "observer_model_total_variation_distance": observer_total_variation_distance,

            "performer_model_distribution_overlap": performer_distribution_overlap,
            "observer_model_distribution_overlap": observer_distribution_overlap,

            "performer_model_logits_standard_deviation": performer_model_logits_standard_deviation,
            "observer_model_logits_standard_deviation": observer_model_logits_standard_deviation,

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
    # These are all "private" to the class and are usually not meant to be directly used (except for maybe debugging)
    # --------------------------------------------------------------------------------------

    
    
    def _compute_telescope_perplexity(
            self,
            text_encoding: BatchEncoding,
            logits: torch.Tensor,
            median: bool = False,
            temperature: float = 1.0
        ) -> float:
        """
        Computes Telescope perplexity on a single sample.

        Args:
            text_encoding (BatchEncoding): Squeezed text encodings containing input_ids and attention_mask.
            logits (torch.Tensor): Squeezed logits from the performer model of shape [sequence_length, vocab_size].
            median (bool, optional): Whether to use the median instead of the mean loss. Defaults to False.
            temperature (float, optional): Logit scaling factor. Defaults to 1.0.

        Returns:
            float: The computed Telescope perplexity score.
        """

        shifted_logits: torch.Tensor = logits[:-1, :].contiguous() / temperature
        shifted_labels: torch.Tensor = text_encoding.input_ids[:-1].contiguous()
        shifted_attention_mask: torch.Tensor = text_encoding.attention_mask[:-1].contiguous()

        cross_entropy_losses: torch.Tensor = cross_entropy_loss_function(shifted_logits, shifted_labels)

        telescope_perplexity: torch.Tensor
        if median:
            valid_cross_entropy: torch.Tensor = cross_entropy_losses[shifted_attention_mask.bool()]
            telescope_perplexity = valid_cross_entropy.median()
        else:
            telescope_perplexity = (cross_entropy_losses * shifted_attention_mask).sum() / shifted_attention_mask.sum()

        return float(telescope_perplexity.item())


    def _compute_perplexity(
            self,
            text_encoding: BatchEncoding,
            logits: torch.Tensor,
            median: bool = False,
            temperature: float = 1.0
        ) -> float:
        """
        Computes standard perplexity on a single sample.

        Args:
            text_encoding (BatchEncoding): Squeezed text encodings containing input_ids and attention_mask.
            logits (torch.Tensor): Squeezed logits from the performer model of shape [sequence_length, vocab_size].
            median (bool, optional): Whether to use the median instead of the mean loss. Defaults to False.
            temperature (float, optional): Logit scaling factor. Defaults to 1.0.

        Returns:
            float: The computed standard perplexity score.
        """
        # from the original Binoculars paper:
        # https://github.com/ahans30/Binoculars
        # https://arxiv.org/pdf/2401.12070
        # Copyright (c) 2023, Abhimanyu Hans, Avi Schwarzschild, Tom Goldstein

        shifted_logits: torch.Tensor = logits[:-1, :].contiguous() / temperature
        shifted_labels: torch.Tensor = text_encoding.input_ids[1:].contiguous()
        shifted_attention_mask: torch.Tensor = text_encoding.attention_mask[1:].contiguous()

        cross_entropy_losses: torch.Tensor = cross_entropy_loss_function(shifted_logits, shifted_labels)

        perplexity: torch.Tensor
        if median:
            valid_cross_entropy: torch.Tensor = cross_entropy_losses[shifted_attention_mask.bool()]
            perplexity = valid_cross_entropy.median()
        else:
            perplexity = (cross_entropy_losses * shifted_attention_mask).sum() / shifted_attention_mask.sum()

        return float(perplexity.item())


    def _compute_cross_perplexity(
            self,
            text_encoding: BatchEncoding,
            p_logits: torch.Tensor,
            q_logits: torch.Tensor,
            pad_token_id: Optional[int],
            median: bool = False,
            sample_p: bool = False,
            temperature: float = 1.0
        ) -> float:
        """
        Computes cross-perplexity (contrastive perplexity) on a single sample.

        Args:
            text_encoding (BatchEncoding): Squeezed text encodings containing input_ids and attention_mask.
            p_logits (torch.Tensor): Squeezed logits from the observer model of shape [sequence_length, vocab_size].
            q_logits (torch.Tensor): Squeezed logits from the performer model of shape [sequence_length, vocab_size].
            pad_token_id (int): Token ID used for padding.
            median (bool, optional): Whether to use the median instead of the mean loss. Defaults to False.
            sample_p (bool, optional): Whether to sample from the model distribution P. Defaults to False.
            temperature (float, optional): Logit scaling factor. Defaults to 1.0.

        Returns:
            float: The computed cross perplexity score.
        """
        # from the original Binoculars paper:
        # https://github.com/ahans30/Binoculars
        # https://arxiv.org/pdf/2401.12070
        # Copyright (c) 2023, Abhimanyu Hans, Avi Schwarzschild, Tom Goldstein

        vocab_size: int = p_logits.shape[-1]
        p_scores: torch.Tensor
        q_scores: torch.Tensor
        p_scores, q_scores = p_logits / temperature, q_logits / temperature

        p_proba: torch.Tensor = softmax_function(p_scores)

        if sample_p:
            p_proba = torch.multinomial(p_proba, replacement=True, num_samples=1).view(-1)

        cross_entropy_losses: torch.Tensor = cross_entropy_loss_function(input=q_scores, target=p_proba)
        padding_mask: torch.Tensor = (text_encoding.input_ids != pad_token_id).type(torch.uint8)

        aggregated_cross_entropy: torch.Tensor
        if median:
            valid_cross_entropy: torch.Tensor = cross_entropy_losses[padding_mask.bool()]
            aggregated_cross_entropy = valid_cross_entropy.median()
        else:
            aggregated_cross_entropy = (cross_entropy_losses * padding_mask).sum() / padding_mask.sum()

        return float(aggregated_cross_entropy.item())


    def _compute_log_rank_ratio(self, text_encoding: BatchEncoding, logits: torch.Tensor) -> float:
        """
        Computes the DetectLLM Log Rank Ratio.
        
        Args:
            text_encoding (BatchEncoding): Squeezed text encodings.
            logits (torch.Tensor): Squeezed logits of shape [sequence_length, vocab_size].
            
        Returns:
            float: Log rank ratio value.
        """
        log_probs: torch.Tensor
        ranks: torch.Tensor
        log_probs, ranks = self._compute_token_probabilities_and_ranks(text_encoding, logits)
        log_ranks: torch.Tensor = torch.log(ranks.float())

        # Take absolute ratio as defined in the paper
        return float(abs(log_probs.sum() / (log_ranks.sum() + 1e-6)).item())


    def _compute_fast_detectgpt(
            self, 
            text_encoding: BatchEncoding,
            logits_ref: torch.Tensor, 
            logits_score: torch.Tensor
        ) -> float:
        """
        Computes the Fast-DetectGPT metric: fast_detectgpt (analytic discrepancy).

        NOTE: This is only the analytic version of Fast-DetectGPT.
        For the sampling-based non-analytic version, see _compute_fast_detectgpt_non_analytic_debug.

        Args:
            text_encoding (BatchEncoding): Squeezed text encodings.
            logits_ref (torch.Tensor): Squeezed reference model logits of shape [sequence_length, vocab_size].
            logits_score (torch.Tensor): Squeezed scoring model logits of shape [sequence_length, vocab_size].

        Returns:
            float: The analytic Fast-DetectGPT score.
        """
        # Slice logits to align with predictions (last token doesn't predict anything)
        logits_ref = logits_ref[:-1]
        logits_score = logits_score[:-1]

        # Ensure shapes match
        if logits_ref.size(-1) != logits_score.size(-1):
            vocab_size: int = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :vocab_size]
            logits_score = logits_score[:, :vocab_size]

        lprobs_score: torch.Tensor = torch.log_softmax(logits_score, dim=-1)
        probs_ref: torch.Tensor = torch.softmax(logits_ref, dim=-1)

        labels: torch.Tensor = text_encoding.input_ids[1:]
        labels_expanded: torch.Tensor = labels.unsqueeze(-1)
        log_likelihood: torch.Tensor = lprobs_score.gather(dim=-1, index=labels_expanded).squeeze(-1)

        mean_ref: torch.Tensor = (probs_ref * lprobs_score).sum(dim=-1)
        var_ref: torch.Tensor = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)

        fast_detectgpt: torch.Tensor = (log_likelihood.sum() - mean_ref.sum()) / var_ref.sum().sqrt()

        return float(fast_detectgpt.item())






    def _compute_total_variation_distance(
            self, 
            text_encoding: BatchEncoding,
            logits: torch.Tensor
        ) -> float:
        """
        Compute total variation distance between the model's predictions and actual next tokens.

        Args:
            text_encoding (BatchEncoding): Squeezed text encodings.
            logits (torch.Tensor): Squeezed logits of shape [sequence_length, vocab_size].

        Returns:
            float: Average total variation distance.
        """
        probs: torch.Tensor = torch.softmax(logits, dim=-1)[:-1]

        next_tokens: torch.Tensor = text_encoding.input_ids[1:]
        actual_probs: torch.Tensor = torch.zeros_like(probs)
        actual_probs[torch.arange(next_tokens.size(0)), next_tokens] = 1

        abs_diff: torch.Tensor = torch.abs(probs - actual_probs)
        token_tv: torch.Tensor = 0.5 * torch.sum(abs_diff, dim=-1)

        mask: torch.Tensor = text_encoding.attention_mask[1:]
        average_total_variation: torch.Tensor = (token_tv * mask).sum() / mask.sum().float()

        return float(average_total_variation.item())


    def _compute_distribution_overlap(
            self, 
            text_encoding: BatchEncoding,
            logits: torch.Tensor
        ) -> float:
        """
        Compute overlap across all token position distributions simultaneously.

        Args:
            text_encoding (BatchEncoding): Squeezed text encodings.
            logits (torch.Tensor): Squeezed logits of shape [sequence_length, vocab_size].

        Returns:
            float: Total distribution overlap value.
        """
        probs: torch.Tensor = torch.softmax(logits, dim=-1)[:-1]

        next_tokens: torch.Tensor = text_encoding.input_ids[1:]
        actual_probs: torch.Tensor = torch.zeros_like(probs)
        actual_probs[torch.arange(next_tokens.size(0)), next_tokens] = 1

        mask: torch.Tensor = text_encoding.attention_mask[1:].unsqueeze(-1)
        probs = probs * mask
        actual_probs = actual_probs * mask

        overlap: torch.Tensor = torch.minimum(probs, actual_probs)
        position_overlap: torch.Tensor = torch.sum(overlap, dim=-1)
        total_overlap: torch.Tensor = position_overlap.sum() / mask.sum().float()

        return float(total_overlap.item())



    def _compute_logits_standard_deviation(
            self, 
            text_encoding: BatchEncoding,
            logits: torch.Tensor
        ) -> float:
        """
        Compute standard deviation of predicted token probability distributions.

        Args:
            text_encoding (BatchEncoding): Squeezed text encodings.
            logits (torch.Tensor): Squeezed logits of shape [sequence_length, vocab_size].

        Returns:
            float: Logits standard deviation value.
        """
        probs: torch.Tensor = torch.softmax(logits, dim=-1)
        masked_probs: torch.Tensor = probs * text_encoding.attention_mask.unsqueeze(-1)
        std: torch.Tensor = torch.std(masked_probs)
        return float(std.item())


    
    def _compute_token_probabilities_and_ranks(
            self, 
            text_encoding: BatchEncoding, 
            logits: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes token log probabilities and ranks on GPU.

        Args:
            text_encoding (BatchEncoding): Squeezed text encodings.
            logits (torch.Tensor): Squeezed logits of shape [sequence_length, vocab_size].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - log_probs (torch.Tensor): 1D log probability tensor.
                - ranks (torch.Tensor): 1D rank tensor.
        """
        logits = logits.to(torch.float32)
        probs: torch.Tensor = torch.softmax(logits, dim=-1)
        
        target_ids: torch.Tensor = text_encoding.input_ids[1:-1]
        eval_probs: torch.Tensor = probs[:-2, :]
        
        token_probs: torch.Tensor = eval_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        log_probs: torch.Tensor = torch.log(token_probs.clamp(min=1e-10))
        
        ranks: torch.Tensor = (eval_probs > token_probs.unsqueeze(-1)).sum(dim=-1) + 1
        
        return log_probs, ranks


    def _compute_log_likelihood(self, text_encoding: BatchEncoding, logits: torch.Tensor) -> float:
        """
        Computes average log likelihood of the sequence.

        Args:
            text_encoding (BatchEncoding): Squeezed text encodings.
            logits (torch.Tensor): Squeezed logits of shape [sequence_length, vocab_size].

        Returns:
            float: Average log likelihood.
        """
        log_probs: torch.Tensor
        ranks: torch.Tensor
        log_probs, ranks = self._compute_token_probabilities_and_ranks(text_encoding, logits)
        return float((log_probs.sum() / (log_probs.numel() + 1e-6)).item())


    def _compute_log_rank(self, text_encoding: BatchEncoding, logits: torch.Tensor) -> float:
        """
        Computes average log rank of the sequence.

        Args:
            text_encoding (BatchEncoding): Squeezed text encodings.
            logits (torch.Tensor): Squeezed logits of shape [sequence_length, vocab_size].

        Returns:
            float: Average log rank.
        """
        log_probs: torch.Tensor
        ranks: torch.Tensor
        log_probs, ranks = self._compute_token_probabilities_and_ranks(text_encoding, logits)
        log_ranks: torch.Tensor = torch.log(ranks.float())
        return float((log_ranks.sum() / (log_ranks.numel() + 1e-6)).item())








    def _compute_telescope_perplexity_per_token(
            self,
            text_encodings: BatchEncoding,
            logits: torch.Tensor,
            temperature: float = 1.0
        ) -> List[float]:
        """
        Computes Telescope perplexity per token.

        Args:
            text_encodings (BatchEncoding): Squeezed text encodings.
            logits (torch.Tensor): Squeezed logits of shape [sequence_length, vocab_size].
            temperature (float, optional): Logit scaling factor. Defaults to 1.0.

        Returns:
            List[float]: A list of Telescope perplexities for each token.
        """
        shifted_logits: torch.Tensor = logits[:-1, :].contiguous() / temperature
        shifted_labels: torch.Tensor = text_encodings.input_ids[:-1].contiguous()
        shifted_attention_mask: torch.Tensor = text_encodings.attention_mask[:-1].contiguous()

        token_loss: torch.Tensor = cross_entropy_loss_function(shifted_logits, shifted_labels)
        valid_losses: torch.Tensor = token_loss[shifted_attention_mask.bool()]
        return valid_losses.tolist()


    def _compute_perplexity_per_token(
            self,
            text_encodings: BatchEncoding,
            logits: torch.Tensor,
            temperature: float = 1.0
        ) -> List[float]:
        """
        Computes standard perplexity per token.

        Args:
            text_encodings (BatchEncoding): Squeezed text encodings.
            logits (torch.Tensor): Squeezed logits of shape [sequence_length, vocab_size].
            temperature (float, optional): Logit scaling factor. Defaults to 1.0.

        Returns:
            List[float]: A list of standard perplexities for each token.
        """
        shifted_logits: torch.Tensor = logits[:-1, :].contiguous() / temperature
        shifted_labels: torch.Tensor = text_encodings.input_ids[1:].contiguous()
        shifted_attention_mask: torch.Tensor = text_encodings.attention_mask[1:].contiguous()

        token_loss: torch.Tensor = cross_entropy_loss_function(shifted_logits, shifted_labels)
        valid_losses: torch.Tensor = token_loss[shifted_attention_mask.bool()]
        return valid_losses.tolist()


    def _compute_cross_perplexity_per_token(
            self,
            text_encodings: BatchEncoding,
            performer_model_logits: torch.Tensor,
            observer_model_logits: torch.Tensor,
            pad_token_id: Optional[int],
            temperature: float = 1.0
        ) -> List[float]:
        """
        Computes cross perplexity per token.

        Args:
            text_encodings (BatchEncoding): Squeezed text encodings.
            performer_model_logits (torch.Tensor): Squeezed performer logits of shape [sequence_length, vocab_size].
            observer_model_logits (torch.Tensor): Squeezed observer logits of shape [sequence_length, vocab_size].
            pad_token_id (int): Token ID used for padding.
            temperature (float, optional): Logit scaling factor. Defaults to 1.0.

        Returns:
            List[float]: A list of cross perplexities for each token.
        """
        vocab_size: int = observer_model_logits.shape[-1]
        p_scores: torch.Tensor
        q_scores: torch.Tensor
        p_scores, q_scores = observer_model_logits / temperature, performer_model_logits / temperature

        p_proba: torch.Tensor = softmax_function(p_scores)
        cross_entropy: torch.Tensor = cross_entropy_loss_function(input=q_scores, target=p_proba)
        
        padding_mask: torch.Tensor = (text_encodings.input_ids != pad_token_id)
        valid_cross_entropy: torch.Tensor = cross_entropy[padding_mask]

        return valid_cross_entropy.tolist()


    def _compute_fast_detectgpt_non_analytic_debug(
            self, 
            text_encoding: BatchEncoding,
            logits_ref: torch.Tensor, 
            logits_score: torch.Tensor
        ) -> float:
        """
        DEBUG ONLY: Computes the non-analytic (sample-based) version of Fast-DetectGPT.
        This is not used in the production calculation flow.

        Args:
            text_encoding (BatchEncoding): Squeezed text encodings.
            logits_ref (torch.Tensor): Squeezed reference model logits of shape [sequence_length, vocab_size].
            logits_score (torch.Tensor): Squeezed scoring model logits of shape [sequence_length, vocab_size].

        Returns:
            float: The non-analytic Fast-DetectGPT score.
        """
        # Ensure shapes match
        if logits_ref.size(-1) != logits_score.size(-1):
            vocab_size: int = min(logits_ref.size(-1), logits_score.size(-1))
            logits_ref = logits_ref[:, :vocab_size]
            logits_score = logits_score[:, :vocab_size]

        lprobs_score: torch.Tensor = torch.log_softmax(logits_score, dim=-1)
        labels: torch.Tensor = text_encoding.input_ids[1:]
        labels_expanded: torch.Tensor = labels.unsqueeze(-1)
        log_likelihood: torch.Tensor = lprobs_score.gather(dim=-1, index=labels_expanded).squeeze(-1)

        nsamples: int = 10000
        lprobs_ref: torch.Tensor = torch.log_softmax(logits_ref, dim=-1)
        distrib: torch.distributions.categorical.Categorical = torch.distributions.categorical.Categorical(logits=lprobs_ref)
        samples: torch.Tensor = distrib.sample([nsamples]).transpose(0, 1)

        log_likelihood_x_samples: torch.Tensor = lprobs_score.gather(dim=-1, index=samples)
        log_likelihood_x_tilde: torch.Tensor = log_likelihood_x_samples.mean(dim=0)

        miu_tilde: torch.Tensor = log_likelihood_x_tilde.mean(dim=-1)
        sigma_tilde: torch.Tensor = log_likelihood_x_tilde.std(dim=-1)

        log_likelihood_x: torch.Tensor = log_likelihood.mean(dim=0)

        discrepancy: torch.Tensor = (log_likelihood_x - miu_tilde) / sigma_tilde

        return float(discrepancy.item())


    def _compute_telescope_perplexity_and_cross_perplexity_from_logits(
            self,
            text_encodings: BatchEncoding,
            performer_model_logits: torch.Tensor,
            observer_model_logits: torch.Tensor,
            reference_offset: int = 0,
            number_of_tokens_to_skip: int = 20,
            device: Optional[Union[str, torch.device]] = None
        ) -> Tuple[float, float]:
        """
        DEBUG ONLY: This is an alternative implementation of the telescope formula designed for debugging/ablation purposes.
        It is designed to be easier to understand and easier to reason through.

        Args:
            text_encodings (BatchEncoding): Squeezed text encodings.
            performer_model_logits (torch.Tensor): Squeezed performer logits of shape [sequence_length, vocab_size].
            observer_model_logits (torch.Tensor): Squeezed observer logits of shape [sequence_length, vocab_size].
            reference_offset (int, optional): Logit offset parameter. Defaults to 0.
            number_of_tokens_to_skip (int, optional): Tokens to skip in sequence start. Defaults to 20.
            device (torch.device, optional): Calculation device. Defaults to None.

        Returns:
            Tuple[float, float]: Telescope perplexity and cross perplexity scores.
        """
        if device is None:
            device = self.device

        observer_model_logits = observer_model_logits.to(device)
        performer_model_logits = performer_model_logits.to(device)
        
        text_input_ids: torch.Tensor = text_encodings["input_ids"].to(device)

        total_cross_entropy_cross_perplexity: torch.Tensor = torch.tensor(0.0, device=device)
        total_cross_entropy_normal_perplexity: torch.Tensor = torch.tensor(0.0, device=device)
        tokens_evaluated: int = 0

        current_token_index: int
        for current_token_index in range(performer_model_logits.shape[0] - reference_offset):
            if current_token_index < number_of_tokens_to_skip: continue

            tokens_evaluated += 1

            performer_next_token_logits: torch.Tensor = performer_model_logits[current_token_index, :].reshape(1, -1)
            observer_next_token_logits: torch.Tensor = observer_model_logits[current_token_index, :].reshape(1, -1)

            performer_next_tokens_logits_softmax: torch.Tensor = torch.softmax(performer_next_token_logits, dim=-1)
            observer_next_token_logits_softmax: torch.Tensor = torch.softmax(observer_next_token_logits, dim=-1)

            total_cross_entropy_cross_perplexity -= torch.matmul(performer_next_tokens_logits_softmax, torch.log(observer_next_token_logits_softmax).T).squeeze()
            total_cross_entropy_normal_perplexity -= torch.log(performer_next_tokens_logits_softmax[0, text_input_ids[current_token_index + reference_offset]])

        normal_perplexity: float
        cross_perplexity: float
        if tokens_evaluated > 0:
            normal_perplexity = float((total_cross_entropy_normal_perplexity / tokens_evaluated).item())
            cross_perplexity = float((total_cross_entropy_cross_perplexity / tokens_evaluated).item())
        else:
            normal_perplexity = 0.0
            cross_perplexity = 0.0

        return normal_perplexity, cross_perplexity


        observer_model_logits = observer_model_logits.to(device)
        performer_model_logits = performer_model_logits.to(device)
        
        text_input_ids: torch.Tensor = text_encodings["input_ids"].to(device)

        total_cross_entropy_cross_perplexity = torch.tensor(0.0, device=device)
        total_cross_entropy_normal_perplexity = torch.tensor(0.0, device=device)
        tokens_evaluated = 0

        for current_token_index in range(performer_model_logits.shape[0] - reference_offset):
            if current_token_index < number_of_tokens_to_skip: continue

            tokens_evaluated += 1

            performer_next_token_logits = performer_model_logits[current_token_index, :].reshape(1, -1)
            observer_next_token_logits = observer_model_logits[current_token_index, :].reshape(1, -1)

            performer_next_tokens_logits_softmax = torch.softmax(performer_next_token_logits, dim=-1)
            observer_next_token_logits_softmax = torch.softmax(observer_next_token_logits, dim=-1)

            total_cross_entropy_cross_perplexity -= torch.matmul(performer_next_tokens_logits_softmax, torch.log(observer_next_token_logits_softmax).T).squeeze()
            total_cross_entropy_normal_perplexity -= torch.log(performer_next_tokens_logits_softmax[0, text_input_ids[current_token_index + reference_offset]])

        if tokens_evaluated > 0:
            normal_perplexity = float((total_cross_entropy_normal_perplexity / tokens_evaluated).item())
            cross_perplexity = float((total_cross_entropy_cross_perplexity / tokens_evaluated).item())
        else:
            normal_perplexity = 0.0
            cross_perplexity = 0.0

        return normal_perplexity, cross_perplexity