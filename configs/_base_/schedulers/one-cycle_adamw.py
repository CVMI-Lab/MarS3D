epochs = 100
start_epoch = 0
optimizer = dict(type='AdamW', lr=0.005, weight_decay=0.0001)
scheduler = dict(type='OneCycleLR',
                 max_lr=optimizer["lr"],
                 epochs=epochs,
                 steps_per_epoch=1,
                 pct_start=0.05,
                 anneal_strategy="cos",
                 div_factor=10.0,
                 final_div_factor=100.0)

