import torch
import numpy as np
import transformers
from torch.nn.functional import cross_entropy as ce_loss_fn
import matplotlib.pyplot as plt
from typing import List, Tuple
import seaborn as sns

def telescope_perplexity(
    encoding: transformers.BatchEncoding,
    logits: torch.Tensor,
    median: bool = False,
    temperature: float = 1.0
    ):

    shifted_logits = logits[..., :-1, :].contiguous() / temperature
    shifted_labels = encoding.input_ids[..., :-1].contiguous()
    shifted_attention_mask = encoding.attention_mask[..., :-1].contiguous()
    if median:
        ce_nan = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels).
                  masked_fill(~shifted_attention_mask.bool(), float("nan")))
        ppl = np.nanmedian(ce_nan.cpu().float().numpy(), 1)
    else:
        ppl = (ce_loss_fn(shifted_logits.transpose(1, 2), shifted_labels) *
               shifted_attention_mask).sum(1) / shifted_attention_mask.sum(1)
        ppl = ppl.to("cpu").float().numpy()
    return ppl

def get_token_probabilities(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    text: str
) -> Tuple[torch.Tensor, transformers.BatchEncoding, torch.Tensor]:
    """Get probability distributions for each position in the text."""
    encoding = tokenizer(text, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**encoding)
    
    # Apply softmax to get probabilities
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs, encoding, outputs.logits

def analyze_distribution_uniformity(
    human_texts: List[str],
    ai_texts: List[str],
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer
) -> Tuple[float, float]:
    """Test hypothesis A: Distribution uniformity comparison"""
    
    def get_avg_entropy(texts):
        entropies = []
        for text in texts:
            probs, _, _ = get_token_probabilities(model, tokenizer, text)
            # Calculate entropy for each position
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            entropies.append(entropy.mean().item())
        return np.mean(entropies)
    
    human_entropy = get_avg_entropy(human_texts)
    ai_entropy = get_avg_entropy(ai_texts)
    
    return human_entropy, ai_entropy

def analyze_next_token_bump(
    human_texts: List[str],
    ai_texts: List[str],
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer
) -> Tuple[float, float]:
    """Test hypothesis B: Next token probability bump"""
    
    def get_avg_next_token_prob(texts):
        next_token_probs = []
        for text in texts:
            probs, encoding, _ = get_token_probabilities(model, tokenizer, text)
            # Get probabilities of actual next tokens
            next_tokens = encoding.input_ids[0, 1:]  # Shift left by 1
            batch_indices = torch.arange(next_tokens.shape[0])
            next_token_prob = probs[0, :-1, next_tokens]
            next_token_probs.append(next_token_prob.mean().item())
        return np.mean(next_token_probs)
    
    human_next_prob = get_avg_next_token_prob(human_texts)
    ai_next_prob = get_avg_next_token_prob(ai_texts)
    
    return human_next_prob, ai_next_prob

def analyze_probability_ratios(
    human_texts: List[str],
    ai_texts: List[str],
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer
) -> dict:
    """Analyze how current tokens appear in next position's probability distribution"""
    
    def get_probability_stats(texts):
        all_current_token_ratios = []
        all_current_token_ratios_no_peak = []
        
        for text in texts:
            probs, encoding, _ = get_token_probabilities(model, tokenizer, text)
            current_tokens = encoding.input_ids[0, :-1]  # All except last
            
            # For each position in the sequence (except last)
            for pos in range(len(current_tokens)):
                # Get probability distribution for next position
                next_pos_probs = probs[0, pos+1]
                current_token = current_tokens[pos]  # The current token we're looking at
                
                # Get probability assigned to current token in next position
                current_token_in_next = next_pos_probs[current_token].item()
                
                # Calculate average probability (including peak)
                avg_prob = torch.mean(next_pos_probs).item()
                ratio = current_token_in_next / avg_prob
                all_current_token_ratios.append(ratio)
                
                # Calculate average excluding the peak (top probability)
                top_prob, top_idx = torch.max(next_pos_probs, dim=0)
                sum_no_peak = torch.sum(next_pos_probs) - top_prob
                count_no_peak = len(next_pos_probs) - 1
                avg_no_peak = sum_no_peak / count_no_peak
                ratio_no_peak = current_token_in_next / avg_no_peak.item()
                all_current_token_ratios_no_peak.append(ratio_no_peak)
        
        return {
            'with_peak': np.mean(all_current_token_ratios),
            'without_peak': np.mean(all_current_token_ratios_no_peak)
        }
    
    human_stats = get_probability_stats(human_texts)
    ai_stats = get_probability_stats(ai_texts)
    
    return {
        'human': human_stats,
        'ai': ai_stats
    }

def analyze_peak_exclusion_ratio(
    human_texts: List[str],
    ai_texts: List[str],
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer
) -> dict:
    """
    Analyzes texts using a peak exclusion ratio metric that compares the distribution
    of token probabilities excluding the highest probability token.
    
    Returns a score where higher values indicate likely AI-generated text.
    """
    def get_peak_exclusion_stats(texts):
        position_ratios = []
        
        for text in texts:
            probs, encoding, _ = get_token_probabilities(model, tokenizer, text)
            current_tokens = encoding.input_ids[0, :-1]
            
            for pos in range(len(current_tokens)):
                next_pos_probs = probs[0, pos+1]
                current_token = current_tokens[pos]
                
                # Get probability for current token in next position
                current_token_prob = next_pos_probs[current_token].item()
                
                # Calculate distribution stats excluding peak
                top_prob, _ = torch.max(next_pos_probs, dim=0)
                sum_no_peak = torch.sum(next_pos_probs) - top_prob
                count_no_peak = len(next_pos_probs) - 1
                avg_no_peak = sum_no_peak / count_no_peak
                
                # Calculate ratio and add position-specific features
                ratio = current_token_prob / avg_no_peak.item()
                
                # Add smoothing to handle very small denominators
                ratio = min(ratio, 1e6)  # Cap extremely large ratios
                
                position_ratios.append(ratio)
        
        return {
            'mean_ratio': np.mean(position_ratios),
            'median_ratio': np.median(position_ratios),
            'std_ratio': np.std(position_ratios),
            'max_ratio': np.max(position_ratios),
            'min_ratio': np.min(position_ratios)
        }
    
    human_stats = get_peak_exclusion_stats(human_texts)
    ai_stats = get_peak_exclusion_stats(ai_texts)
    
    # Calculate composite score based on statistical features
    def calculate_detection_score(stats):
        return (np.log1p(stats['mean_ratio']) * 
                np.log1p(stats['max_ratio']) * 
                (1 + stats['std_ratio']))
    
    results = {
        'human': {
            'stats': human_stats,
            'detection_score': calculate_detection_score(human_stats)
        },
        'ai': {
            'stats': ai_stats,
            'detection_score': calculate_detection_score(ai_stats)
        }
    }
    
    return results

def calculate_perplexities(
    human_texts: List[str],
    ai_texts: List[str],
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer
) -> Tuple[float, float]:
    """Calculate telescope perplexity for both human and AI texts"""
    
    def get_avg_perplexity(texts):
        perplexities = []
        for text in texts:
            _, encoding, logits = get_token_probabilities(model, tokenizer, text)
            ppl = telescope_perplexity(encoding, logits)
            perplexities.extend(ppl)
        return np.mean(perplexities)
    
    human_ppl = get_avg_perplexity(human_texts)
    ai_ppl = get_avg_perplexity(ai_texts)
    
    return human_ppl, ai_ppl

def visualize_results(
    human_entropy: float,
    ai_entropy: float,
    human_next_prob: float,
    ai_next_prob: float,
    human_ppl: float,
    ai_ppl: float,
    ratios: dict,
    peak_exclusion_results: dict
):
    """Visualize all results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot entropy comparison
    entropy_data = {
        'Source': ['Human', 'AI'],
        'Entropy': [human_entropy, ai_entropy]
    }
    sns.barplot(x='Source', y='Entropy', data=entropy_data, ax=ax1)
    ax1.set_title('Distribution Uniformity (Entropy)')
    
    # Plot next token probability comparison
    prob_data = {
        'Source': ['Human', 'AI'],
        'Next Token Probability': [human_next_prob, ai_next_prob]
    }
    sns.barplot(x='Source', y='Next Token Probability', data=prob_data, ax=ax2)
    ax2.set_title('Next Token Probability')
    
    # Plot perplexity comparison
    ppl_data = {
        'Source': ['Human', 'AI'],
        'Telescope Perplexity': [human_ppl, ai_ppl]
    }
    sns.barplot(x='Source', y='Telescope Perplexity', data=ppl_data, ax=ax3)
    ax3.set_title('Telescope Perplexity')
    
    # Plot detection scores from peak exclusion analysis
    score_data = {
        'Source': ['Human', 'AI'],
        'Detection Score': [
            peak_exclusion_results['human']['detection_score'],
            peak_exclusion_results['ai']['detection_score']
        ]
    }
    sns.barplot(x='Source', y='Detection Score', data=score_data, ax=ax4)
    ax4.set_title('Peak Exclusion Detection Score')
    ax4.set_yscale('log')
    
    plt.tight_layout()
    return fig

def visualize_peak_exclusion_analysis(results: dict) -> None:
    """Visualize the peak exclusion analysis results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot mean ratios
    mean_data = {
        'Source': ['Human', 'AI'],
        'Mean Ratio': [results['human']['stats']['mean_ratio'],
                      results['ai']['stats']['mean_ratio']]
    }
    sns.barplot(x='Source', y='Mean Ratio', data=mean_data, ax=ax1)
    ax1.set_title('Mean Peak Exclusion Ratio')
    ax1.set_yscale('log')
    
    # Plot standard deviations
    std_data = {
        'Source': ['Human', 'AI'],
        'Std Ratio': [results['human']['stats']['std_ratio'],
                     results['ai']['stats']['std_ratio']]
    }
    sns.barplot(x='Source', y='Std Ratio', data=std_data, ax=ax2)
    ax2.set_title('Standard Deviation of Ratios')
    ax2.set_yscale('log')
    
    # Plot max ratios
    max_data = {
        'Source': ['Human', 'AI'],
        'Max Ratio': [results['human']['stats']['max_ratio'],
                     results['ai']['stats']['max_ratio']]
    }
    sns.barplot(x='Source', y='Max Ratio', data=max_data, ax=ax3)
    ax3.set_title('Maximum Ratio')
    ax3.set_yscale('log')
    
    # Plot detection scores
    score_data = {
        'Source': ['Human', 'AI'],
        'Detection Score': [results['human']['detection_score'],
                          results['ai']['detection_score']]
    }
    sns.barplot(x='Source', y='Detection Score', data=score_data, ax=ax4)
    ax4.set_title('Overall Detection Score')
    ax4.set_yscale('log')
    
    plt.tight_layout()
    return fig
def main():
    # Initialize model and tokenizer
    model_name = "HuggingFaceTB/SmolLM-360M-Instruct"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    
    human_texts = [
        '''
Hi, Dr. Hamouda!

I'm one of your former 3114 students (I took it last semester).


Myself and a colleague have developed a zero-shot algorithm for accurately detecting LLM generated text (including code). We are currently finishing up experiments and writing a paper for submission to NerIPS, with assistance from Dr. Jin from the ECE department.

We are very interested in seeing how it performs in a real world environment (ideally rigorously). 

Would you be interested in integrating it with 3114's Webcat, or some other use case you have as an educator that we could collaborate on?

I've included a results table for some of the testing we have done, the previous state of the art model is called Binoculars, and E5-small is a trained detector being tested on out of distribution data.
'''
    ]
    
    ai_texts = [
        '''Dear Dr. Hamouda,

I hope you're doing well! I was a student in your 3114 course last semester, and I wanted to reach out regarding a project my colleague and I have been working on.

We've developed a zero-shot algorithm capable of accurately detecting AI-generated text, including code. As we finalize our experiments and prepare our paper for submission to NeurIPS—with guidance from Dr. Jin in the ECE department—we are eager to evaluate its performance in real-world applications.

Given your experience as an educator, we were wondering if you'd be interested in collaborating to integrate our model into Webcat or exploring other potential use cases within 3114. We believe this could provide valuable insights into the effectiveness of AI detection in academic settings.

I've attached a results table from our testing, comparing our model against existing state-of-the-art approaches, including Binoculars and E5-small on out-of-distribution data. I'd love to discuss this further if you're interested.

Looking forward to your thoughts!'''
    ]
    
    # Run all analyses
    human_entropy, ai_entropy = analyze_distribution_uniformity(
        human_texts, ai_texts, model, tokenizer
    )
    
    human_next_prob, ai_next_prob = analyze_next_token_bump(
        human_texts, ai_texts, model, tokenizer
    )
    
    human_ppl, ai_ppl = calculate_perplexities(
        human_texts, ai_texts, model, tokenizer
    )
    
    ratios = analyze_probability_ratios(
        human_texts, ai_texts, model, tokenizer
    )
    
    # Calculate peak exclusion results
    peak_exclusion_results = analyze_peak_exclusion_ratio(
        human_texts, ai_texts, model, tokenizer
    )
    
    # Print all results
    print("\nHypothesis A - Distribution Uniformity:")
    print(f"Human text entropy: {human_entropy:.4f}")
    print(f"AI text entropy: {ai_entropy:.4f}")
    
    print("\nHypothesis B - Next Token Probability:")
    print(f"Human text next token prob: {human_next_prob:.4f}")
    print(f"AI text next token prob: {ai_next_prob:.4f}")
    
    print("\nTelescope Perplexity:")
    print(f"Human text perplexity: {human_ppl:.4f}")
    print(f"AI text perplexity: {ai_ppl:.4f}")
    
    print("\nNext Token Probability Ratios:")
    print("\nHuman Text:")
    print(f"Relative to full distribution average: {ratios['human']['with_peak']:.4f}x")
    print(f"Relative to average excluding peak: {ratios['human']['without_peak']:.4f}x")
    print("\nAI Text:")
    print(f"Relative to full distribution average: {ratios['ai']['with_peak']:.4f}x")
    print(f"Relative to average excluding peak: {ratios['ai']['without_peak']:.4f}x")
    
    print("\nPeak Exclusion Analysis:")
    print("\nHuman Text:")
    print(f"Mean Ratio: {peak_exclusion_results['human']['stats']['mean_ratio']:.4f}")
    print(f"Detection Score: {peak_exclusion_results['human']['detection_score']:.4f}")
    print("\nAI Text:")
    print(f"Mean Ratio: {peak_exclusion_results['ai']['stats']['mean_ratio']:.4f}")
    print(f"Detection Score: {peak_exclusion_results['ai']['detection_score']:.4f}")
    
    # Visualize results
    fig = visualize_results(
        human_entropy, ai_entropy,
        human_next_prob, ai_next_prob,
        human_ppl, ai_ppl,
        ratios,
        peak_exclusion_results
    )
    plt.savefig('all_results.png')
    
    # Create separate visualization for peak exclusion analysis
    fig2 = visualize_peak_exclusion_analysis(peak_exclusion_results)
    plt.savefig('peak_exclusion_analysis.png')

if __name__ == "__main__":
    main()