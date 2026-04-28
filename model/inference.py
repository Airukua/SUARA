import torch
import torch.nn.functional as F

def build_generation_case(texts, prompt_words=12, target_words=24):
    for text in texts:
        words = text.split()
        if len(words) > prompt_words + 4:
            prompt = ' '.join(words[:prompt_words])
            target = ' '.join(words[prompt_words:prompt_words + target_words])
            return prompt, target
    fallback_prompt = "the history of"
    return fallback_prompt, ""

@torch.no_grad()
def generate_sample(model, tokenizer, device, prompt, MAX_SEQ=None, max_new_tokens=40,
                    temperature=0.9, top_k=40, ):
    model = model.to(device)
    model.eval()
    if MAX_SEQ is None:
        MAX_SEQ = getattr(model, 'max_seq', 128)
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        token_ids = [tokenizer.bos_token_id]

    generated = list(token_ids)

    for _ in range(max_new_tokens):
        ctx = generated[-MAX_SEQ:]
        inp = torch.tensor([ctx], dtype=torch.long, device=device)
        logits, _ = model(inp)
        next_logits = logits[0, -1] / max(temperature, 1e-5)

        if top_k is not None and top_k > 0:
            k = min(top_k, next_logits.size(-1))
            top_vals, top_idx = torch.topk(next_logits, k)
            probs = F.softmax(top_vals, dim=-1)
            next_token = top_idx[torch.multinomial(probs, 1)].item()
        else:
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated)
