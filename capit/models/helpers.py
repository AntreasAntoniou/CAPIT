import torch

from capit.base import utils

log = utils.get_logger(__name__)


def contrastive_logits_labels(logits: torch.Tensor):
    # logit shape is expected to be (batch_size, num_classes)
    labels = torch.arange(logits.shape[1]).type_as(logits).long()
    return logits, labels


def check_modality_embedding_shape(
    input_shape, embedding_name, modality_embeddings, embed_dim
):
    input_dummy = torch.zeros(size=input_shape)

    embeddings, _ = modality_embeddings[embedding_name].forward(input_dummy)

    assert embeddings.shape[1] == embed_dim


def get_normalized_features(inputs, embedding_name, modality_embeddings):

    if inputs is not None:
        inputs, _ = modality_embeddings[embedding_name](inputs)
        inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return inputs
    else:
        return None


def compute_cross_modal_cosine_similarities(
    embedding_dict, logit_scale_params, logit_scale_dict
):
    logit_dict = {}
    for source_key, source_value in embedding_dict.items():
        for target_key, target_value in embedding_dict.items():
            if (
                source_key != target_key
                and source_value is not None
                and target_value is not None
            ):
                if f"{source_key}_to_{target_key}" in logit_scale_dict:
                    logit_scale_idx = logit_scale_dict[f"{source_key}_to_{target_key}"]
                else:
                    logit_scale_idx = logit_scale_dict[f"{target_key}_to_{source_key}"]

                logit_scale = logit_scale_params[logit_scale_idx].exp()

                if f"{target_key}_to_{source_key}_similarity" in logit_dict:
                    logit_dict[f"{source_key}_to_{target_key}_similarity"] = logit_dict[
                        f"{target_key}_to_{source_key}_similarity"
                    ].t()
                else:
                    logit_dict[f"{source_key}_to_{target_key}_similarity"] = (
                        torch.matmul(source_value, target_value.t()) * logit_scale
                    )

    return logit_dict
