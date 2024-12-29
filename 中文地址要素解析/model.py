import torch
from torchcrf import CRF
from transformers import AutoModelForTokenClassification

class TokenClassificationWithCRF(torch.nn.Module):
    def __init__(self, base_model_path, num_labels):
        super(TokenClassificationWithCRF, self).__init__()
        self.num_labels = num_labels
        self.base_model = AutoModelForTokenClassification.from_pretrained(base_model_path, num_labels=num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]  # [batch_size, seq_len, num_labels]

        if labels is not None:
            loss = -self.crf(logits, labels, mask=attention_mask.byte())
            return {"loss": loss}
        else:
            predictions = self.crf.decode(logits, mask=attention_mask.byte())
            return {"logits": predictions}
