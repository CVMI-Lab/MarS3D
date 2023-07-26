epochs = 100
start_epoch = 0
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.0001)
scheduler = dict(type='CosineAnnealingLR', epochs=epochs, steps_per_epoch=1, eta_min=0.0001)
