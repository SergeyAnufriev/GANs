import wandb


sweep_config = {'program':'lightening2.py','method':'grid','parameters':{'n_hidden_generator':{'values':[10,50,200]},\
                                              'n_hidden_discriminator':{'values':[10,50,200]}, \
                                              'activation':{'values':['LeakyReLU', 'Sigmoid', 'Tanh']}}}


sweep_id = wandb.sweep(sweep_config, entity="zcemg08", project="gpu_try2")


text_file = open('/home/zcemg08/Scratch/GANs/gpu_runs/id.txt', "w")
text_file.write(sweep_id)
text_file.close()


print (sweep_id)






