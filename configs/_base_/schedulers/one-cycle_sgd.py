epochs = 100
start_epoch = 0
optimizer = dict(type='SGD', lr=0.5, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = dict(type='OneCycleLR',
                 max_lr=optimizer["lr"],
                 epochs=epochs,
                 steps_per_epoch=1,
                 pct_start=0.05,
                 anneal_strategy="cos",
                 div_factor=10.0,
                 final_div_factor=10000.0)
