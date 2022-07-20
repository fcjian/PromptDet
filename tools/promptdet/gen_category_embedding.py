import argparse
import clip
import torch
import torch.nn as nn
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Laion images of the LVIS novel categories to mmdetection format')
    parser.add_argument('--model-file', help='the model weight of the regional prompt learning')
    parser.add_argument('--name-file', default='promptdet_resources/prompt_learner/lvis/category_and_description.txt',
                        help='the category name and description')
    parser.add_argument('--out-file', default='promptdet_resources/lvis_category_embeddings.pt',
                        help='output path')
    args = parser.parse_args()
    return args


class PromptLearner(nn.Module):
    def __init__(self, n_ctx, category_descriptions, clip_model):
        super().__init__()
        classnames = [category_description.split("_which_is_")[0] for category_description in category_descriptions]
        classdefs = [category_description.split("_which_is_")[1] for category_description in category_descriptions]

        n_cls = len(classnames)
        n_ctx1 = n_ctx
        n_ctx2 = n_ctx
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # random initialization
        print("Initializing a generic context")
        ctx_vectors1 = torch.empty(n_ctx1, ctx_dim, dtype=dtype)
        ctx_vectors2 = torch.empty(n_ctx2, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors1, std=0.02)
        nn.init.normal_(ctx_vectors2, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx1)
        prompt_suffix = " ".join(["X"] * n_ctx2)

        print(f'Initial context1: "{prompt_prefix}"')
        print(f'Initial context2: "{prompt_suffix}"')
        print(f"Number of context words (tokens1): {n_ctx1}")
        print(f"Number of context words (tokens2): {n_ctx2}")

        self.ctx1 = nn.Parameter(ctx_vectors1)  # to be optimized
        self.ctx2 = nn.Parameter(ctx_vectors2)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        classdefs = [des.replace("_", " ") for des in classdefs]
        def_lens = [len(_tokenizer.encode(name)) for name in classdefs]

        prompts = [prompt_prefix + " " + name + " " + prompt_suffix + " " + classdef + "." for name, classdef in
                   zip(classnames, classdefs)]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(
            next(clip_model.parameters()).device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("embedding", embedding)  # SOS

        self.n_cls = n_cls
        self.n_ctx1 = n_ctx1
        self.n_ctx2 = n_ctx2
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.def_lens = def_lens

    def forward(self):
        ctx1 = self.ctx1
        ctx2 = self.ctx2
        if ctx1.dim() == 2:
            ctx1 = ctx1.unsqueeze(0).expand(self.n_cls, -1, -1)
            ctx2 = ctx2.unsqueeze(0).expand(self.n_cls, -1, -1)

        embedding = self.embedding

        n_ctx1 = self.n_ctx1
        n_ctx2 = self.n_ctx2
        prompts = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            def_len = self.def_lens[i]
            prefix_i = embedding[i: i + 1, :1, :]
            ctx1_i = ctx1[i:i + 1]
            name_index = 1 + n_ctx1
            name_i = embedding[i: i + 1, name_index:name_index + name_len, :]
            ctx2_i = ctx2[i:i + 1]
            def_index = 1 + n_ctx1 + name_len + n_ctx2
            def_i = embedding[i: i + 1, def_index:def_index + def_len, :]
            suf_index = 1 + n_ctx1 + name_len + n_ctx2 + def_len
            suffix_i = embedding[i: i + 1, suf_index:, :]
            prompt = torch.cat(
                [
                    prefix_i,
                    ctx1_i,
                    name_i,
                    ctx2_i,
                    def_i,
                    suffix_i,
                ],
                dim=1,
            )
            prompts.append(prompt)
        prompts = torch.cat(prompts, dim=0)

        return prompts


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


def main():
    args = parse_args()

    model_file = args.model_file # 'promptdet_resources/prompt_learner/lvis/model.pth.tar-6'
    name_file = args.name_file # "promptdet_resources/prompt_learner/lvis/category_and_description.txt"
    out_file = args.out_file # "promptdet_resources/lvis_category_embeddings_gen.pt"

    lines = open(name_file).readlines()
    names = [line.strip().split(' ')[-1] for line in lines]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    for name, param in model.named_parameters():
        param.requires_grad = False

    prompt_learner = PromptLearner(1, names, model)
    text_encoder = TextEncoder(model)
    model_dict = torch.load(model_file)
    prompt_learner.ctx1.data = model_dict['state_dict']['ctx1']
    prompt_learner.ctx2.data = model_dict['state_dict']['ctx2']

    prompts = prompt_learner()
    tokenized_prompts = prompt_learner.tokenized_prompts
    text_features = text_encoder(prompts, tokenized_prompts)
    print(f'The shape of the category embedding: {text_features.shape}')

    torch.save(text_features.cpu().detach(), out_file)
    print('ok!')


if __name__ == '__main__':
    main()
