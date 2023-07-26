epochs = 100
start_epoch = 0
optimizer = dict(type='AdamW', lr=0.005, weight_decay=0.05)
scheduler = dict(type='MultiStepLR', milestones=[epochs * 0.6, epochs * 0.8], steps_per_epoch=1, gamma=0.1)
