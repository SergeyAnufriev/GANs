# GANs

lightening2.py is the main code

name.py generates w&b agent id to run 

run everything with torch_bash1.sh

(edit files inside for your user name and project name) 


To run code:

1) pip install -r requirements.txt  

2) edit gan_torch.py: 

sweep_id = wandb.sweep(sweep_config, entity="zcemg08", project="gans_training2")

a) entity: your weights and biases username 

b) project: name of your project (create in weights and biases web site) 

3) edit torch_bash1.sh

a) Use your own weights and biases API key

b) Change files location 

4) run torch_bash1.sh 





