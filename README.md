# GANs

To run code:

1) pip install -r requirements.txt  

edit gan_torch.py: 

2) sweep_id = wandb.sweep(sweep_config, entity="zcemg08", project="gans_training2")

a) entity: your weights and biases username 

b) project: name of your project (create in weights and biases web site) 

3) edit torch_bash.sh

a) Use your own weights and biases API key

b) Change your own files location 

4) run torch_bash.sh 





