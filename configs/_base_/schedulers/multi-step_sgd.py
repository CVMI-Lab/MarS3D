epochs = 100
start_epoch = 0
optimizer = dict(type='SGD', lr=0.5, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = dict(type='MultiStepLR', milestones=[epochs * 0.6, epochs * 0.8], steps_per_epoch=1, gamma=0.1)
