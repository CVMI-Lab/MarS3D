epochs = 100
start_epoch = 0
optimizer = dict(type='AdamW', lr=0.006, weight_decay=0.05)
scheduler = dict(type='MultiStepWithWarmupLR', milestones=[epochs * 0.6, epochs * 0.8],
                 steps_per_epoch=1, gamma=0.1, warmup_epochs=3, warmup_ratio=1e-6)

