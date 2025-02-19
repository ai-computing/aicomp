#
# Copyright (c) 2025-present, ETRI, All rights reserved.
#
#   This code is simply intended for running the vision transformer in vanilla PyTorch
#           and has been adapted from the following source:
#
#              https://github.com/NielsRogge/Transformers-Tutorials/blob/master/VisionTransformer/Fine_tuning_the_Vision_Transformer_on_CIFAR_10_with_the_%F0%9F%A4%97_Trainer.ipynb 
#
#

from datasets import load_dataset

# load cifar10 (only small portion)
#train_ds, test_ds = load_dataset('cifar10', split=['train[:5000]', 'test[:2000]'])
train_ds = load_dataset('cifar10', split=['train[:5000]'])[0]

print(f"> train_ds: {train_ds}")
print(f"> train_ds.features: {train_ds.features}")
print(f"> train_ds[0]['label']: {train_ds[0]['label']}")

id2label = {id:label for id, label in enumerate(train_ds.features['label'].names)}
label2id = {label:id for id,label in id2label.items()}

# id2label -> { 0: 'airplane', 1: 'automobile', 2: 'bird', ..., 9: 'truck'}
# Usage: print(id2label[train_ds[0]['label']])

from transformers import ViTImageProcessor

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomHorizontalFlip,
                                    RandomResizedCrop,
                                    Resize,
                                    ToTensor)

image_mean, image_std = processor.image_mean, processor.image_std
size = processor.size["height"]

normalize = Normalize(mean=image_mean, std=image_std)
_train_transforms = Compose(
        [
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ]
    )

def train_transforms(examples):
    examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
    return examples


train_ds.set_transform(train_transforms)

#print(train_ds[:2])


from torch.utils.data import DataLoader
import torch

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

dataloader = DataLoader(train_ds, collate_fn=collate_fn, batch_size=4)


batch = next(iter(dataloader))
for k,v in batch.items():
  if isinstance(v, torch.Tensor):
    print(k, v.shape)

from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=10, id2label=id2label, label2id=label2id)

print(model)

import torch.optim as optim

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        #print(f"pixel_values: {pixel_values}, labels: {v}")
        pixel_values, labels = pixel_values.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(pixel_values).logits

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')
