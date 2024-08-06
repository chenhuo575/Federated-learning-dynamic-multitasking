from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar

class Global_models:
    def __init__(self,global_models=[],global_weights=[],train_loss_s=[],train_accuracys=[],val_acc_lists=[],net_lists=[],cv_loss_s=[],cv_accs=[]):
        self.global_models=global_models
        self.global_weights=global_weights
        self.train_loss_s=train_loss_s
        self.train_accuracys=train_accuracys
        self.val_acc_lists=val_acc_lists
        self.net_lists=net_lists
        self.cv_loss_s=cv_loss_s
        self.cv_accs=cv_accs

    def set_models(self,device):
        for global_model in self.global_models:
            global_model.to(device)
            global_model.train()
            print(global_model)

            # copy weights
            self.global_weights.append(global_model.state_dict())
            
            self.train_loss_s.append([]) 
            self.train_accuracys.append([])
            self.val_acc_lists.append([])
            self.net_lists.append([])
            self.cv_loss_s.append([])
            self.cv_accs.append([])
        return "set_models_successfully"
    
    def add_model(self,args,train_dataset):
        if args.model == 'cnn':
        # Convolutional neural netork
            if args.dataset == 'mnist':
                self.global_models.append(CNNMnist(args=args))
            elif args.dataset == 'fmnist':
                self.global_models.append(CNNFashion_Mnist(args=args))
            elif args.dataset == 'cifar':
                self.global_models.append(CNNCifar(args=args))
        elif args.model == 'mlp':
                # Multi-layer preceptron
            img_size = train_dataset[0][0].shape
            len_in = 1
            for x in img_size:
                len_in *= x
                cur_model = MLP(dim_in=len_in, dim_hidden=64,
                                dim_out=args.num_classes)
            self.global_models.append(cur_model)
        else:
            exit('Error: unrecognized model')
        return "add_successfully"

    def set_model(self,device,i):
        self.global_models[i].to(device)
        self.global_models[i].train()
        print(self.global_models[i])

        # copy weights
        self.global_weights.append(self.global_models[i].state_dict())
        
        self.train_loss_s.append([]) 
        self.train_accuracys.append([])
        self.val_acc_lists.append([])
        self.net_lists.append([])
        self.cv_loss_s.append([])
        self.cv_accs.append([])

        return "set_model_successfully"