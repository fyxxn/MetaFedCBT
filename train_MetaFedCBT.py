import pickle
import random

import numpy as np

from meta_model import MetaDGN,MetaMLP,MetaUNet
from helper import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_meta_model_optimizer_dict(number_of_samples,fold):
    '''
    This function creates dictionaries that contain optimize function, loss function, gnn model for every device introduced
    to the federated pipeline.
    '''
    model_dict = dict()
    optimizer_dict = dict()
    criterion_dict = dict()
    MLP_dict=dict()
    PCA_dict=dict()
    rdgn_dict=dict()

    for i in range(number_of_samples):
        model_name = "model" + str(i)
        model_info = MetaDGN(MODEL_PARAMS).to(device)
        model_dict.update({model_name: model_info})

        if isSimulate:
            rdgn_name='rdgn'+str(i)
            rdgn_info=MetaUNet(1,N_views).to(device)
            rdgn_info.eval()
            rdgn_info.load_state_dict(torch.load( '{}fold{}_rdgn_cli_{}_{}.model'.format(Path_output_rdgn, fold, fold, i), map_location=device))
            rdgn_dict.update({rdgn_name:rdgn_info})


        optimizer_name = "optimizer" + str(i)
        optimizer_info = torch.optim.AdamW(model_info.parameters(), lr=MODEL_PARAMS["learning_rate"], weight_decay=0.00)
        optimizer_dict.update({optimizer_name: optimizer_info})

        criterion_name = "criterion" + str(i)
        criterion_info = []
        criterion_dict.update({criterion_name: criterion_info})

        MLP_name = "MLP" + str(i)
        setup_seed(87)
        MLP_info = MetaMLP(8225, 5484, 3600,2)
        MLP_dict.update({MLP_name: MLP_info})

        PCA_name="PCA"+str(i)
        PCA_info=PCA(n_components=10)
        PCA_dict.update({PCA_name:PCA_info})

    return model_dict, optimizer_dict, criterion_dict,MLP_dict,PCA_dict,rdgn_dict

def meta_sync_global_to_local(main_model, model_dict, name_of_models, \
                                                   number_of_samples, clients_with_access,one=False):
    '''
    This function updates clients with the newly global model so that all the devices can take advantage of the common
    information.
    '''
    with torch.no_grad():
        if not one:
            for i in range(number_of_samples):  # clients_with_access

                model_dict[name_of_models[i]].conv1.nn[0].weight.data = main_model.conv1.nn[0].weight.data.clone()
                model_dict[name_of_models[i]].conv1.nn[0].bias.data = main_model.conv1.nn[0].bias.data.clone()
                model_dict[name_of_models[i]].conv1.bias.data = main_model.conv1.bias.data.clone()
                model_dict[name_of_models[i]].conv1.lin.weight.data = main_model.conv1.lin.weight.data.clone()

                model_dict[name_of_models[i]].conv2.nn[0].weight.data = main_model.conv2.nn[0].weight.data.clone()
                model_dict[name_of_models[i]].conv2.nn[0].bias.data = main_model.conv2.nn[0].bias.data.clone()
                model_dict[name_of_models[i]].conv2.bias.data = main_model.conv2.bias.data.clone()
                model_dict[name_of_models[i]].conv2.lin.weight.data = main_model.conv2.lin.weight.data.clone()

                model_dict[name_of_models[i]].conv3.nn[0].weight.data = main_model.conv3.nn[0].weight.data.clone()
                model_dict[name_of_models[i]].conv3.nn[0].bias.data = main_model.conv3.nn[0].bias.data.clone()
                model_dict[name_of_models[i]].conv3.bias.data = main_model.conv3.bias.data.clone()
                model_dict[name_of_models[i]].conv3.lin.weight.data = main_model.conv3.lin.weight.data.clone()
        else:
            for i in range(number_of_samples):  # clients_with_access

                model_dict.conv1.nn[0].weight.data = main_model.conv1.nn[0].weight.data.clone()
                model_dict.conv1.nn[0].bias.data = main_model.conv1.nn[0].bias.data.clone()
                model_dict.conv1.bias.data = main_model.conv1.bias.data.clone()
                model_dict.conv1.lin.weight.data = main_model.conv1.lin.weight.data.clone()

                model_dict.conv2.nn[0].weight.data = main_model.conv2.nn[0].weight.data.clone()
                model_dict.conv2.nn[0].bias.data = main_model.conv2.nn[0].bias.data.clone()
                model_dict.conv2.bias.data = main_model.conv2.bias.data.clone()
                model_dict.conv2.lin.weight.data = main_model.conv2.lin.weight.data.clone()

                model_dict.conv3.nn[0].weight.data = main_model.conv3.nn[0].weight.data.clone()
                model_dict.conv3.nn[0].bias.data = main_model.conv3.nn[0].bias.data.clone()
                model_dict.conv3.bias.data = main_model.conv3.bias.data.clone()
                model_dict.conv3.lin.weight.data = main_model.conv3.lin.weight.data.clone()

    return model_dict

def meta_get_averaged_weights(model_dict, name_of_models, number_of_samples, clients_with_access, num_local,
                         average_all=True, last_updated_dict=None, current_epoch=-1):
    '''
    This function averages model weights after a designated number of round so that we can have the weights of the global model
    that takes full advantage of introduced devices in the federated pipeline.
    '''
    conv1_nn_mean_weight = torch.zeros(size=model_dict[name_of_models[0]].conv1.nn[0].weight.data.shape).to(device)
    conv1_nn_mean_bias = torch.zeros(size=model_dict[name_of_models[0]].conv1.nn[0].bias.data.shape).to(device)
    conv1_bias = torch.zeros(size=model_dict[name_of_models[0]].conv1.bias.data.shape).to(device)
    conv1_root = torch.zeros(size=model_dict[name_of_models[0]].conv1.lin.weight.data.shape).to(device)

    conv2_nn_mean_weight = torch.zeros(size=model_dict[name_of_models[0]].conv2.nn[0].weight.data.shape).to(device)
    conv2_nn_mean_bias = torch.zeros(size=model_dict[name_of_models[0]].conv2.nn[0].bias.data.shape).to(device)
    conv2_bias = torch.zeros(size=model_dict[name_of_models[0]].conv2.bias.data.shape).to(device)
    conv2_root = torch.zeros(size=model_dict[name_of_models[0]].conv2.lin.weight.data.shape).to(device)

    conv3_nn_mean_weight = torch.zeros(size=model_dict[name_of_models[0]].conv3.nn[0].weight.data.shape).to(device)
    conv3_nn_mean_bias = torch.zeros(size=model_dict[name_of_models[0]].conv3.nn[0].bias.data.shape).to(device)
    conv3_bias = torch.zeros(size=model_dict[name_of_models[0]].conv3.bias.data.shape).to(device)
    conv3_root = torch.zeros(size=model_dict[name_of_models[0]].conv3.lin.weight.data.shape).to(device)

    if average_all:
        cls = range(number_of_samples)
    else:
        cls = clients_with_access
    num_local_cls = [num_local[i] for i in cls] # part num

    with torch.no_grad():
        # def getWeight_i(i):
        #     return ((np.e / 2) ** (- (current_epoch - last_updated_dict['client' + str(i)])))

        for i in cls:  # cls
            if 'avg' in Setup_name:
                client_weight = 1 / len(cls)
            # elif istemp:
            #     client_weight=getWeight_i(i)/(sum([getWeight_i(n) for n in range(number_of_samples)]))
            else:
                client_weight = num_local[i] / sum(num_local_cls)
            conv1_nn_mean_weight += (client_weight * model_dict[name_of_models[i]].conv1.nn[0].weight.data.clone())
            conv1_nn_mean_bias += (client_weight * model_dict[name_of_models[i]].conv1.nn[0].bias.data.clone())
            conv1_bias += (client_weight * model_dict[name_of_models[i]].conv1.bias.data.clone())
            conv1_root += (client_weight * model_dict[name_of_models[i]].conv1.lin.weight.data.clone())

            conv2_nn_mean_weight += (client_weight * model_dict[name_of_models[i]].conv2.nn[0].weight.data.clone())
            conv2_nn_mean_bias += (client_weight * model_dict[name_of_models[i]].conv2.nn[0].bias.data.clone())
            conv2_bias += (client_weight * model_dict[name_of_models[i]].conv2.bias.data.clone())
            conv2_root += (client_weight * model_dict[name_of_models[i]].conv2.lin.weight.data.clone())

            conv3_nn_mean_weight += (client_weight * model_dict[name_of_models[i]].conv3.nn[0].weight.data.clone())
            conv3_nn_mean_bias += (client_weight * model_dict[name_of_models[i]].conv3.nn[0].bias.data.clone())
            conv3_bias += (client_weight * model_dict[name_of_models[i]].conv3.bias.data.clone())
            conv3_root += (client_weight * model_dict[name_of_models[i]].conv3.lin.weight.data.clone())

    return conv1_nn_mean_weight, conv1_nn_mean_bias, conv1_bias, conv1_root, \
           conv2_nn_mean_weight, conv2_nn_mean_bias, conv2_bias, conv2_root, \
           conv3_nn_mean_weight, conv3_nn_mean_bias, conv3_bias, conv3_root

def meta_set_global_model_weights(main_model, model_dict, name_of_models,
                                                                     number_of_samples, clients_with_access,current_epoch, num_local):
    '''
    This function takes combined weights for global model and assigns them to the global model.
    '''
    conv1_nn_mean_weight, conv1_nn_mean_bias, conv1_bias, conv1_root, \
    conv2_nn_mean_weight, conv2_nn_mean_bias, conv2_bias, conv2_root, \
    conv3_nn_mean_weight, conv3_nn_mean_bias, conv3_bias, conv3_root = meta_get_averaged_weights(model_dict, \
                                                                                            name_of_models,
                                                                                            number_of_samples,
                                                                                            clients_with_access,
                                                                                            num_local,
                                                                                            average_all,
                                                                                            current_epoch)

    with torch.no_grad():
        main_model.conv1.nn[0].weight.data = conv1_nn_mean_weight.clone()
        main_model.conv1.nn[0].bias.data = conv1_nn_mean_bias.clone()
        main_model.conv1.bias.data = conv1_bias.clone()
        main_model.conv1.lin.weight.data = conv1_root.clone()

        main_model.conv2.nn[0].weight.data = conv2_nn_mean_weight.clone()
        main_model.conv2.nn[0].bias.data = conv2_nn_mean_bias.clone()
        main_model.conv2.bias.data = conv2_bias.clone()
        main_model.conv2.lin.weight.data = conv2_root.clone()

        main_model.conv3.nn[0].weight.data = conv3_nn_mean_weight.clone()
        main_model.conv3.nn[0].bias.data = conv3_nn_mean_bias.clone()
        main_model.conv3.bias.data = conv3_bias.clone()
        main_model.conv3.lin.weight.data = conv3_root.clone()

    return main_model


def meta_fed_train_kfold(Path_input, loss_table_list, number_of_samples):
    '''
    This function is the main loop. Executes all the functions mentioned above and federated pipeline.
    '''
    for i in range(n_folds):
        result_avg_rep = dict()
        result_avg_top = dict()
        for j_th in range(number_of_samples):
            result_avg_rep[j_th] = []
            result_avg_top[j_th] = []
        for random_count in range(random_num):

            loss_table = dict()
            for m in range(number_of_samples):
                loss_table.update({'local_loss_global_data_' + str(m): []})
                loss_table.update({'combining_local_loss_global_data_' + str(m): []})

            client_access = [True] * number_of_samples
            torch.cuda.empty_cache()

            all_train_data = np.load('{}fold{}/fold{}_train_data.npy'.format(Path_input, i, i))
            all_test_data = np.load('{}fold{}/fold{}_test_data.npy'.format(Path_input, i, i))
            all_train_data, all_test_data = map(torch.tensor, (all_train_data, all_test_data))
            a_file = open("{}fold{}/client_data_fold_{}.pkl".format(Path_input, i, i), "rb")
            x_train_dict = pickle.load(a_file)
            a_file.close()

            # dictionaries are ready to be involved into the federated pipeline.
            model_dict, optimizer_dict, criterion_dict,MLP_dict,PCA_dict,rdgn_dict = create_meta_model_optimizer_dict(number_of_samples,i)
            name_of_x_train_sets = list(x_train_dict.keys())
            name_of_models = list(model_dict.keys())
            name_of_optimizers = list(optimizer_dict.keys())
            name_of_criterions = list(criterion_dict.keys())
            name_of_MLP=list(MLP_dict.keys())
            name_of_PCA=list(PCA_dict.keys())
            name_of_rdgn=list(rdgn_dict.keys())

            #load CBT center
            if isSimulate:
                CBT_true=dict()
                CBT_true_all = dict()
                for  m in range(number_of_samples):
                    CBT_true.update({name_of_x_train_sets[m]:np.load('{}fold{}_0_cli_{}_{}_all_cbts.npy'.format(Path_output_CBT, i, i,m)).transpose(2,0,1)})
                    CBT_true_all.update({name_of_x_train_sets[m]: np.load('{}fold{}_0_cli_{}_{}_cbt.npy'.format(Path_output_CBT, i, i, m))})

            # main model creation
            setup_seed(SEED)
            main_model = MetaDGN(MODEL_PARAMS)
            main_model = main_model.to(device)
            test_casted = [d.to(device) for d in cast_data(all_test_data)]
            meta_sync_global_to_local(main_model, model_dict, name_of_models,number_of_samples, list(range(0, number_of_samples)))
            num_local = [len(x_train_dict[name_of_x_train_sets[ith_model]]) for ith_model in range(number_of_samples)]

            #visualization
            X_data =np.concatenate([Vectorize(x_train_dict[name_of_x_train_sets[ith_model]]) for ith_model in range(number_of_samples)])
            Y_data = np.concatenate([[i for x in range(num_local[i])] for i in range(number_of_samples)])
            # plotTSNE(X_data, Y_data, len(name_of_x_train_sets), i, True, False)
            # plotPCA(X_data, Y_data, len(name_of_x_train_sets), i, True, False)

            for ith_model in range( number_of_samples):  # for every device in the federated pipeline, compute initial validation values
                train_data_ith_model = x_train_dict[name_of_x_train_sets[ith_model]]
                train_casted_ith_model = [d.to(device) for d in cast_data(train_data_ith_model)]
                model_ith = model_dict[name_of_models[ith_model]].to(device)
                kth_model_rep_loss = validation(model_ith,train_data_ith_model, train_casted_ith_model, test_casted)
                print(ith_model, 'th sample loss:', "{:.4f}".format(kth_model_rep_loss), end=' ')

            simulate_train=dict()
            x_train=x_train_dict
            isStart=False
            isConvergence=False
            isbegin=False
            Start_simulate=Start_simulate_num

            for n in range(N_max_epochs):  # every epoch's loop
                client_indices = [i for i, x in enumerate(client_access) if x]
                if len(client_indices) > 1:
                    if int(len(client_indices)*C_sgd) < 1:
                        clients_with_access = sorted(random.sample(client_indices, 1))
                    else:
                        clients_with_access = sorted(random.sample(client_indices, int(len(client_indices)*C_sgd)))
                elif len(client_indices) == 1:
                    if isFederated:
                        break
                    clients_with_access = sorted(random.sample(client_indices, 1))
                else:
                    break
                if isConvergence:
                    Start_simulate=n//numEpoch+1
                    print("isconvergence")
                    print(Start_simulate)
                    isbegin=True
                    isConvergence=False

                if  (n + 1) <= (Start_simulate + 1) * numEpoch:
                    clients_with_access = [i for i in range(number_of_samples)]

                # models learning
                if isMeta and n==Start_simulate*numEpoch:
                    model_list = dict()
                    op_list = dict()
                    for k in range(number_of_samples):
                        model_list[name_of_models[k]]=dict()
                        op_list[name_of_models[k]]=dict()
                        for data in range(len(simulate_train[name_of_x_train_sets[0]])):
                            model_name="model"+str(data)
                            model_info = MetaDGN(MODEL_PARAMS).to(device)
                            model_list[name_of_models[k]].update({model_name: model_info})
                            optimizer_name = "optimizer" + str(data)
                            optimizer_info = torch.optim.AdamW(model_info.parameters(), lr=MODEL_PARAMS["learning_rate"],weight_decay=0.00)
                            optimizer_info.load_state_dict(optimizer_dict[name_of_optimizers[k]].state_dict())
                            op_list[name_of_models[k]].update({optimizer_name: optimizer_info})
                        name_model = list(model_list[name_of_models[k]].keys())
                        meta_sync_global_to_local(model_dict[name_of_models[k]], model_list[name_of_models[k]],name_model,len(simulate_train[name_of_x_train_sets[0]]), clients_with_access)

                if isMeta and n >= Start_simulate * numEpoch and (n+1) <=(Start_simulate+1)*numEpoch:
                    start_train_end_node_process(model_list, criterion_dict, op_list, simulate_train,
                                                 name_of_x_train_sets, name_of_models, name_of_criterions,name_of_optimizers,
                                                 clients_with_access, isStart, True)
                    for k in range(number_of_samples):
                        meta_sync_global_to_local(model_list[name_of_models[k]]['model0'],model_dict[name_of_models[k]],['model0'], 1,clients_with_access,True)
                else:
                    start_train_end_node_process(model_dict, criterion_dict, optimizer_dict, x_train,
                                                 name_of_x_train_sets, name_of_models, name_of_criterions, name_of_optimizers,
                                                 clients_with_access,isStart)

                for ith_model in range(number_of_samples):  # loop of every device of every epoch
                    train_data_ith_model = x_train_dict[name_of_x_train_sets[ith_model]]  # data that the device has
                    train_casted_ith_model = [d.to(device) for d in cast_data(train_data_ith_model)]
                    model_ith = model_dict[name_of_models[ith_model]].to(device)
                    kth_model_rep_loss = validation(model_ith,train_data_ith_model, train_casted_ith_model, test_casted)
                    torch.save(model_ith.state_dict(), TEMP_FOLDER+ "/weight_" + model_id + "_" + str(kth_model_rep_loss)[:5] + "_" + str(ith_model) + ".model")
                    loss_table['local_loss_global_data_' + str(ith_model)].append(kth_model_rep_loss)
                    print(n)
                    print(kth_model_rep_loss)
                    if not isFederated:
                        if len(loss_table[ 'local_loss_global_data_' + str(ith_model)]) > early_stop_limit and early_stop:
                                last_6 = loss_table['local_loss_global_data_' + str(ith_model)][-early_stop_limit:]
                                if (all(last_6[i] <= last_6[i + 1] for i in range(early_stop_limit - 1))):
                                    client_access[ith_model] = False

                if not isConvergence and not isbegin:
                    for ith_model in range(number_of_samples):
                        if len(loss_table['local_loss_global_data_' + str(ith_model)]) > convergence_limit:
                            last_6 = loss_table['local_loss_global_data_' + str(ith_model)][-convergence_limit:]
                            if (all((last_6[i] - last_6[i + 1])<con_begin for i in range(convergence_limit - 1))):
                                isConvergence=True
                            else:
                                isConvergence=False

                if (n + 1) == (numEpoch * Start_simulate) and isSimulate:
                        isStart = True
                        simulate_train,y1,y2=meta_CBT_simulate(rdgn_dict,name_of_rdgn,x_train_dict,name_of_x_train_sets,CBT_true,model_dict,name_of_models,client_access=clients_with_access)

                if isStart and not isMeta:
                    x_train = simulate_train
                if isFederated and (n+1)%numEpoch==0:
                    print(isFederated)
                    print(isLocal)
                    main_model = meta_set_global_model_weights(main_model, model_dict, name_of_models,  number_of_samples, clients_with_access,
                                                                                                  n + 1,
                                                                                                  num_local)
                    #
                    if isMeta  and (n + 1) >= (numEpoch * (Start_simulate+1)):#part
                        if (n + 1) == (numEpoch * (Start_simulate+1)):
                            for k in range(number_of_samples):
                                meta_train_MLP(model_list[name_of_models[k]],list(model_list[name_of_models[k]].keys()),y1[k],y2[k],MLP_dict[name_of_MLP[k]],k)
                        if n%3==0 or (n + 1) == (numEpoch * (Start_simulate+1)):
                            pre_mu,pre_std= meta_test_MLP(model_dict, name_of_models, main_model, clients_with_access,PCA_dict,name_of_PCA)
                            simulate_after_train,y1, y2= meta_CBT_simulate(rdgn_dict,name_of_rdgn,x_train_dict,name_of_x_train_sets,CBT_true,model_dict,name_of_models,pre_mu,pre_std,
                                                                  False,clients_with_access)
                            x_train=simulate_after_train

                    if (n+1)>=(numEpoch * (Start_simulate+1)) and not isLocal:#(for compartion)
                        meta_sync_global_to_local(main_model, model_dict, name_of_models,number_of_samples, clients_with_access)

                    for ith_model in range(number_of_samples):
                        train_data_ith_model = x_train_dict[name_of_x_train_sets[ith_model]]
                        train_casted_ith_model = [d.to(device) for d in cast_data(train_data_ith_model)]
                        model_ith = model_dict[name_of_models[ith_model]].to(device)
                        kth_model_rep_loss=validation(model_ith,train_data_ith_model, train_casted_ith_model, test_casted)
                        print(kth_model_rep_loss)
                        print(n)
                        torch.save(model_ith.state_dict(), TEMP_FOLDER+ "/weight_" + model_id + "_" + str(kth_model_rep_loss)[:5] + "_" + str(ith_model) + ".model")
                        #loss_table['local_loss_global_data_' + str(ith_model)].append(kth_model_rep_loss)
                        loss_table['combining_local_loss_global_data_' + str(ith_model)].append(kth_model_rep_loss)
                        if len(loss_table['combining_local_loss_global_data_' + str(ith_model)]) > early_stop_limit and early_stop:
                            last_6 = loss_table['combining_local_loss_global_data_' + str(ith_model)][-early_stop_limit:]
                            if (all(last_6[i] <= last_6[i + 1] for i in range(early_stop_limit - 1))):
                                client_access[ith_model] = False
            loss_table_list.append(loss_table)
            print(n)
            for k in range( number_of_samples):  # saving models, logging operations etc for every deviced introduced to the federated pipeline.
                cur_model = MetaDGN(MODEL_PARAMS).to(device)
                restore = None
                if isFederated:
                    print(loss_table['combining_local_loss_global_data_' + str(k)].index(min(loss_table['combining_local_loss_global_data_' + str(k)])))
                    restore = TEMP_FOLDER+ "/weight_" + model_id + "_" + str(  min(loss_table['combining_local_loss_global_data_' + str(k)]))[:5] + "_" + str(k) + ".model"
                else:
                    print(loss_table['local_loss_global_data_' + str(k)].index(min(loss_table['local_loss_global_data_' + str(k)])))
                    restore = TEMP_FOLDER+ "/weight_" + model_id + "_" + str(min(loss_table['local_loss_global_data_' + str(k)]))[:5] + "_" + str(k) + ".model"
                cur_model.load_state_dict(torch.load(restore))
                torch.save(cur_model.state_dict(), '{}fold{}_cli_{}_{}_{}.model'.format(Path_output, i, random_count,i, k, Setup_name))

                kth_train_data = x_train_dict[name_of_x_train_sets[k]]
                kth_train_casted = [d.to(device) for d in cast_data(kth_train_data)]

                cbt = meta_generate_cbt_median(cur_model, kth_train_casted)
                rep_loss = mean_frobenious_distance(cbt, test_casted).cpu()
                result_avg_rep[k].append(rep_loss)
                kl_loss = float(sum(KL_error(cbt, test_casted, True)))
                result_avg_top[k].append(kl_loss)
                cbt = cbt.cpu().numpy()
                print(random_count,k, 'th model', 'final loss based on kth_train_casted-cbt:', rep_loss, kl_loss)
                np.save('{}fold{}_{}_cli_{}_{}_cbt'.format(Path_output, i, random_count,i, k, Setup_name), cbt)
                all_cbts = meta_generate_subject_biased_cbts(cur_model, kth_train_casted)
                np.save('{}fold{}_{}_cli_{}_{}_all_cbts'.format(Path_output, i,random_count, i, k, Setup_name), all_cbts)
                with open('{}fold{}_loss'.format(Path_output, i, i), "a+") as file:
                    file.write('Loss for {}  {} Client {} is {} {}\n'.format(Setup_name,random_count, k, rep_loss,kl_loss))
            print('------------------------------End of the fold------------------------------')
            with open('output/' + Dataset_name + '/' + Setup_name + '/loss' + str(i) + '.pkl', 'wb') as handle:
                pickle.dump(loss_table_list, handle)
        for k in range(number_of_samples):
            with open('{}fold{}_avg_loss'.format(Path_output, i, i), "a+") as file:
                file.write('Loss for {}  Simulated Client {} is {} {} {} {}\n'.format(Setup_name,  k, np.array(result_avg_rep[k]).mean(),np.array(result_avg_rep[k]).min(),np.array(result_avg_top[k]).mean(),np.array(result_avg_top[k]).min()))
                # file.write('Loss for {}  Simulated Client {} is {} {} {} {}\n'.format(Setup_name,  k, np.array(result_avg_rep[k]).mean(),np.array(result_avg_rep[k]).min()),np.array(result_avg_top[k]).mean(),np.array(result_avg_top[k]).min())
            print(np.array(result_avg_rep[k]).mean())
            print(np.array(result_avg_rep[k]).min())
            print(np.array(result_avg_top[k]).mean())
            print(np.array(result_avg_top[k]).min())
        print('------------------------------End results of the fold------------------------------')


def validation(model,x_train,train_casted, test_casted):
    '''
    Model is tested with frobenious distance in this function so that we can benchmark.
    '''
    model.eval()
    cbt = meta_generate_cbt_median(model, train_casted)
    rep_loss = mean_frobenious_distance(cbt, test_casted)
    rep_loss = float(rep_loss)
    # (35x35) : shape of cbt
    return rep_loss
def meta_test_MLP(model_local,name_of_model,main_model,client_access,PCA_dict,name_of_PCA):
    pre_mu=[]
    pre_std=[]
    for k in range(number_of_samples):
        #if k in client_access:
        conv1_nn_mean_weight = (model_local[name_of_model[k]].conv1.nn[0].weight.data.clone() -main_model.conv1.nn[0].weight.data.clone()).reshape(-1)
        conv1_nn_mean_bias = (model_local[name_of_model[k]].conv1.nn[0].bias.data.clone() -main_model.conv1.nn[0].bias.data.clone()).reshape(-1)
        conv1_bias = (model_local[name_of_model[k]].conv1.bias.data.clone() - main_model.conv1.bias.data.clone()).reshape(-1)
        conv1_root = (model_local[name_of_model[k]].conv1.lin.weight.data.clone() - main_model.conv1.lin.weight.data.clone()).reshape(-1)
        #
        conv2_nn_mean_weight = (model_local[name_of_model[k]].conv2.nn[0].weight.data.clone() -main_model.conv2.nn[0].weight.data.clone()).reshape(-1)
        conv2_nn_mean_bias = (model_local[name_of_model[k]].conv2.nn[0].bias.data.clone() -main_model.conv2.nn[0].bias.data.clone()).reshape(-1)
        conv2_bias = (model_local[name_of_model[k]].conv2.bias.data.clone() - main_model.conv2.bias.data.clone()).reshape(-1)
        conv2_root = (model_local[name_of_model[k]].conv2.lin.weight.data.clone() - main_model.conv2.lin.weight.data.clone()).reshape(-1)

        conv3_nn_mean_weight = (model_local[name_of_model[k]].conv3.nn[0].weight.data.clone() -main_model.conv3.nn[0].weight.data.clone()).reshape(-1)
        conv3_nn_mean_bias = (model_local[name_of_model[k]].conv3.nn[0].bias.data.clone() -main_model.conv3.nn[0].bias.data.clone()).reshape(-1)
        conv3_bias = (model_local[name_of_model[k]].conv3.bias.data.clone() - main_model.conv3.bias.data.clone()).reshape(-1)
        conv3_root = (model_local[name_of_model[k]].conv3.lin.weight.data.clone() - main_model.conv3.lin.weight.data.clone()).reshape(-1)
        conv_data=torch.cat([conv1_root,conv1_bias,conv1_nn_mean_bias,conv1_nn_mean_weight,conv2_root,conv2_bias,conv2_nn_mean_bias,conv2_nn_mean_weight,conv3_root,conv3_bias,conv3_nn_mean_bias,conv3_nn_mean_weight]).cpu().numpy()
        test_x=torch.tensor(conv_data,dtype=torch.float32).to(device)

        MLP_local= MetaMLP(8225, 5484, 3600,2).to(device)
        MLP_local.load_state_dict(
                            torch.load(
                                TEMP_FOLDER+ "/mlp" + str(k)  + "_fold.model"
                            )
                        )
        MLP_local.eval()
        test_y=MLP_local(test_x).cpu().detach().numpy()
        print("-----test statistic predict:", test_y)
        pre_mu.append([test_y[0]])
        pre_std.append([test_y[1]])
        # else:
        #     print("-----test statistic predict: [0,0]")
        #     pre_mu.append([0])
        #     pre_std.append([0])

    # else:
    #     pre_mean.append([])
    #     pre_std.append([])
    return pre_mu,pre_std

        #get residusl
        #load weights
        #predict


def meta_train_MLP(model_dict,name_of_models,y1,y2,MlP,k):
    #optimizer = torch.optim.SGD(MlP.parameters(), 0.001)
    optimizer = torch.optim.AdamW(MlP.parameters(), 0.001)
    MetaMLP.to(device)
    loss_func = torch.nn.MSELoss()
    train_x=[]
    train_y = []
    for i in range(1,len(name_of_models)):
        local_model=MetaDGN(MODEL_PARAMS)
        name=[name_of_models[i],name_of_models[0]]
        local_model=meta_set_global_model_weights(local_model,model_dict,name,2 , [i for i in range(2)],-1, [1 for i in range(2)])
    # return local_model_list
        conv1_nn_mean_weight = (model_dict[name_of_models[0]].conv1.nn[0].weight.data.clone()-local_model.conv1.nn[0].weight.data.clone()).reshape(-1)
        conv1_nn_mean_bias = (model_dict[name_of_models[0]].conv1.nn[0].bias.data.clone()-local_model.conv1.nn[0].bias.data.clone()).reshape(-1)
        conv1_bias = (model_dict[name_of_models[0]].conv1.bias.data.clone()-local_model.conv1.bias.data.clone()).reshape(-1)
        conv1_root = (model_dict[name_of_models[0]].conv1.lin.weight.data.clone()-local_model.conv1.lin.weight.data.clone()).reshape(-1)

        conv2_nn_mean_weight = (model_dict[name_of_models[0]].conv2.nn[0].weight.data.clone()-local_model.conv2.nn[0].weight.data.clone()).reshape(-1)
        conv2_nn_mean_bias = (model_dict[name_of_models[0]].conv2.nn[0].bias.data.clone()-local_model.conv2.nn[0].bias.data.clone()).reshape(-1)
        conv2_bias = (model_dict[name_of_models[0]].conv2.bias.data.clone()-local_model.conv2.bias.data.clone()).reshape(-1)
        conv2_root = (model_dict[name_of_models[0]].conv2.lin.weight.data.clone()-local_model.conv2.lin.weight.data.clone()).reshape(-1)

        conv3_nn_mean_weight = (model_dict[name_of_models[0]].conv3.nn[0].weight.data.clone()-local_model.conv3.nn[0].weight.data.clone()).reshape(-1)
        conv3_nn_mean_bias = (model_dict[name_of_models[0]].conv3.nn[0].bias.data.clone()-local_model.conv3.nn[0].bias.data.clone()).reshape(-1)
        conv3_bias = (model_dict[name_of_models[0]].conv3.bias.data.clone()-local_model.conv3.bias.data.clone()).reshape(-1)
        conv3_root = (model_dict[name_of_models[0]].conv3.lin.weight.data.clone()-local_model.conv3.lin.weight.data.clone()).reshape(-1)
        conv_data=torch.cat([conv1_root,conv1_bias,conv1_nn_mean_bias,conv1_nn_mean_weight,conv2_root,conv2_bias,conv2_nn_mean_bias,conv2_nn_mean_weight,conv3_root,conv3_bias,conv3_nn_mean_bias,conv3_nn_mean_weight])

        train_x.append(conv_data)
        train_y.append(torch.tensor([y1[i-1],y2[i-1]],dtype=torch.float32).to(device))
    best_loss=float("inf")
    index_x = torch.randperm(len(train_x))
    loss_list=[]
    for epotch in range(max_epochs_MLP):
        MetaMLP.train()
        loss1=0.0
        loss2=0.0
        for i in range(0,int(len(train_x)*0.8)):
            xx=train_x[index_x[i]]
            out=MetaMLP(xx)
            loss =loss_func(out, train_y[index_x[i]])
            loss1+=loss
        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        MetaMLP.eval()
        for i in range(int(len(train_x)*0.8),len(train_x)):
            out=MetaMLP(train_x[index_x[i]])
            loss = loss_func(out, train_y[index_x[i]])
            loss2+=loss
        loss_list.append(loss2)
        if loss2< best_loss:
            best_loss=loss2
            torch.save(MetaMLP.state_dict(),TEMP_FOLDER+ "/mlp" + str(k)  + "_fold.model")
        print("---val_loss--:", loss2)
        if len(loss_list)>early_stop_limit:
            last_6 =loss_list[-early_stop_limit:]
            if (all(last_6[i] < last_6[i + 1] for i in range(early_stop_limit- 1))):
                break
    print("---val_loss--:", loss2)
    print(epotch)
def train(model, train_data_set, targets, losses, optimizer, loss_weightes):
    '''
    Every model is trained with its own dataset in this function.
    '''
    model.train()
    losses = []
    for train_data in train_data_set:
        train_casted = [d.to(device) for d in cast_data(train_data)]
        for data in train_casted:
            cbt = model(data)
            random_sample_size = int(len(train_data) * random_size)
            if len(targets)>len(train_data):
                views_sampled=[]
                for i in range(len(train_data_set)):
                   views_sampled2=random.sample(targets[len(train_data)*i:(i+1)*len(train_data)], random_sample_size)
                   views_sampled.extend(views_sampled2)
                random_sample_size *=len(train_data_set)
            else:
                views_sampled = random.sample(targets, random_sample_size)
            sampled_targets = torch.cat(views_sampled, axis=2).permute((2, 1, 0))
            expanded_cbt = cbt.expand((sampled_targets.shape[0], MODEL_PARAMS["N_ROIs"], MODEL_PARAMS["N_ROIs"]))
            diff = torch.abs(expanded_cbt - sampled_targets)  # Absolute difference
            sum_of_all = torch.mul(diff, diff).sum(axis=(1, 2))  # Sum of squares
            l = torch.sqrt(sum_of_all)  # Square root of the sum
            if istp:
                rep = (l * loss_weightes[:random_sample_size * MODEL_PARAMS["n_attr"]]).mean()
                cbt_dist=cbt.sum(axis=1)
                cbt_probs=cbt_dist/cbt_dist.sum()
                #View 1
                target_mean1= sampled_targets[range(0,random_sample_size*N_views,N_views)].mean(axis=0)
                target_dist1=target_mean1.sum(axis=1)
                target_probs1=target_dist1/target_dist1.sum()
                kl_loss1=((cbt_probs* torch.log2(cbt_probs/target_probs1)).sum().abs())+((target_probs1* torch.log2(target_probs1/cbt_probs)).sum().abs())
                #view 2
                target_mean1 = sampled_targets[range(1, random_sample_size * N_views, N_views)].mean(axis=0)
                target_dist1 = target_mean1.sum(axis=1)
                target_probs1 = target_dist1 / target_dist1.sum()
                kl_loss2 = ((cbt_probs * torch.log2(cbt_probs / target_probs1)).sum().abs()) + ( (target_probs1 * torch.log2(target_probs1 / cbt_probs)).sum().abs())
                #view3
                target_mean1 = sampled_targets[range(2, random_sample_size * N_views, N_views)].mean(axis=0)
                target_dist1 = target_mean1.sum(axis=1)
                target_probs1 = target_dist1 / target_dist1.sum()
                kl_loss3= ((cbt_probs * torch.log2(cbt_probs / target_probs1)).sum().abs()) + (
                    (target_probs1 * torch.log2(target_probs1 / cbt_probs)).sum().abs())
                #view4
                target_mean1 = sampled_targets[range(3, random_sample_size * N_views, N_views)].mean(axis=0)
                target_dist1 = target_mean1.sum(axis=1)
                target_probs1 = target_dist1 / target_dist1.sum()
                kl_loss4 = ((cbt_probs * torch.log2(cbt_probs / target_probs1)).sum().abs()) + (
                    (target_probs1 * torch.log2(target_probs1 / cbt_probs)).sum().abs())
                #view 5
                target_mean1 = sampled_targets[range(4, random_sample_size * N_views, N_views)].mean(axis=0)
                target_dist1 = target_mean1.sum(axis=1)
                target_probs1 = target_dist1 / target_dist1.sum()
                kl_loss5 = ((cbt_probs * torch.log2(cbt_probs / target_probs1)).sum().abs()) + (
                    (target_probs1 * torch.log2(target_probs1 / cbt_probs)).sum().abs())
                #view 6
                target_mean1 = sampled_targets[range(5, random_sample_size * N_views, N_views)].mean(axis=0)
                target_dist1 = target_mean1.sum(axis=1)
                target_probs1 = target_dist1 / target_dist1.sum()
                kl_loss6= ((cbt_probs * torch.log2(cbt_probs / target_probs1)).sum().abs()) + (
                    (target_probs1 * torch.log2(target_probs1 / cbt_probs)).sum().abs())
                kl_loss=(kl_loss6+kl_loss2+kl_loss1+kl_loss5+kl_loss4+kl_loss3)
                losses.append(kl_loss*MODEL_PARAMS["lambda_kl"]+rep)
            else:
                losses.append((l * loss_weightes[:random_sample_size * MODEL_PARAMS["n_attr"]]).sum())
    optimizer.zero_grad()
    loss = torch.mean(torch.stack(losses))
    loss.backward()
    optimizer.step()


def start_train_end_node_process(model_dict, criterion_dict, optimizer_dict,x_train_dict,
                                 name_of_x_train_sets, name_of_models, name_of_criterions,name_of_optimizers,
                                 clients_with_access,is_Start=False,is_MLP=False):
    if is_MLP:
        clients_with_access_train=[i for i in range(number_of_samples)]

    else:
        clients_with_access_train=clients_with_access


    for k in clients_with_access_train:
        if not is_Start :
            train_sets = [x_train_dict[name_of_x_train_sets[k]]]
        else:
            train_sets = x_train_dict[name_of_x_train_sets[k]]
        if is_MLP:
            name_model=list(model_dict[name_of_models[k]].keys())
            name_op=list(optimizer_dict[name_of_models[k]].keys())
            for data in range(len(train_sets)):
                train_data =train_sets[data]
                train_mean = np.mean(train_data, axis=(0, 1, 2))
                loss_weightes = torch.tensor(np.array(list((1 / train_mean) / np.max(1 / train_mean)) * len(train_data)), dtype=torch.float32).to(device)
                model = model_dict[name_of_models[k]][name_model[data]].to(device)
                optimizer = optimizer_dict[name_of_models[k]][name_op[data]]
                loss = criterion_dict[name_of_criterions[k]]
                targets = [torch.tensor(tensor, dtype=torch.float32).to(device) for tensor in train_data]
                train(model, [train_sets[data]], targets, loss, optimizer, loss_weightes)
        else:
            train_data = np.concatenate(train_sets)
            train_mean = np.mean(train_data, axis=(0, 1, 2))
            loss_weightes = torch.tensor(np.array(list((1 / train_mean) / np.max(1 / train_mean)) * len(train_data)*2),dtype=torch.float32).to(device)
            model=model_dict[name_of_models[k]]
            optimizer = optimizer_dict[name_of_optimizers[k]]
            loss = criterion_dict[name_of_criterions[k]]
            targets = [torch.tensor(tensor, dtype=torch.float32).to(device) for tensor in np.concatenate(train_sets)]
            train(model, train_sets, targets, loss, optimizer, loss_weightes)


def meta_CBT_simulate(rdgn_dict,name_of_rdgn,x_train,name_of_train,cbt_center=None,train_model=None,name_of_model=None,mu=None,std=None,isRandom=True,client_access=None):
    simulate_train=dict()
    client_access = [i for i in range(number_of_samples)]
    y1=[]
    y2=[]
    x_client=[]
    y_client=[]
    plot_list=[0]

    for i_index in range(number_of_samples):
        simulate_train[name_of_train[i_index]] = [x_train[name_of_train[i_index]]]
        if i_index in client_access:
            sample_num =len(x_train[name_of_train[i_index]])
            x_data = []
            x_data.extend(Vectorize(x_train[name_of_train[i_index]]))
            y_data = [0 for i in range(sample_num)]
            vec_cbt =np.median(cbt_center[name_of_train[i_index]],axis=0)
            cbt_mean, cbt_std = vec_cbt.mean(), vec_cbt.std()
            print(cbt_mean,cbt_std)

            if isRandom:
                if not isMeta:
                    pre_mu = np.concatenate([np.random.uniform(-bias, bias, 1)])
                    pre_std =np.random.uniform(-bias, bias, 1)
                else:
                    pre_mu = []
                    pre_std = []
                    for i in range(simulate_num):
                        a = np.random.uniform(-bias + (2*bias * i) / simulate_num,-bias + (2*bias * (i + 1)) / simulate_num, 1)
                        for j in range(simulate_num ):
                            b = np.random.uniform(-bias+ (2*bias* j) / (simulate_num ), -bias + (2*bias * (j + 1)) / (simulate_num ), 1)
                            pre_mu.append(a)
                            pre_std.append(b)
                    pre_mu = np.concatenate(pre_mu)
                    pre_std = np.concatenate(pre_std)  # np.random.uniform(-cbt_mean/2,cbt_mean/2, simulate_num)
                print("simulate:", pre_mu[0], pre_std[0])
                print(len(pre_mu))
                print(pre_std)
            else:
                pre_mu=mu[i_index]
                pre_std=std[i_index]

            for num in range(len(pre_mu)):
                new_cbts = []
                noise=sym_noise(pre_mu[num],cbt_std)
                for i in range(sample_num):
                    cbt_torch = torch.tensor(cbt_center[name_of_train[i_index]][i]+noise, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    cbt_torch = (cbt_torch - cbt_torch.mean()) / cbt_torch.std() * abs(cbt_center[name_of_train[i_index]][i].std() + pre_std[num]) + cbt_torch.mean()
                    cbt_torch[cbt_torch<0]=0
                    new_cbt_client=symVectorize(rdgn_dict[name_of_rdgn[i_index]](cbt_torch).squeeze().cpu().detach().numpy(),N_Nodes,vec_cbt)
                    new_cbts.append(new_cbt_client)
                vec_subject = Vectorize(np.array(new_cbts).transpose(0, 3, 2, 1))
                x_data.extend(vec_subject)
                y_data.extend([num + 1 for i in range(sample_num)])
                simulate_train[name_of_train[i_index]].append(np.array(new_cbts).transpose(0,3,2,1))
            # cbts=[]
            # for num in range(len(pre_mu)+1):
            #     train_casted_simulate = [d.to(device) for d in cast_data( simulate_train[name_of_train[i_index]][num])]
            #     cbts.append(meta_generate_cbt_median(train_model[name_of_model[i_index]],train_casted_simulate))

            # kl_list=[]
            # for ith_c in range(len(cbts)):
            #     kl_val = torch.tensor([(torch.mean(torch.nan_to_num(cbts[ith_c] * torch.log2(cbts[ith_c] / cbts[i_c]))) +
            #                             torch.mean(torch.nan_to_num(cbts[i_c] * torch.log2(cbts[i_c] / cbts[ith_c]))))for i_c in range(len(cbts))]).numpy()
            #     kl_list.append(kl_val)
            # kl_matrix(np.array(kl_list))
            # plotTSNE(x_data, y_data, len(pre_mu)+1 , i_index, False,True)
            # plotPCA(x_data, y_data, len(pre_mu)+1 , i_index,False,True)
            y1.append(pre_mu)
            y2.append(pre_std)
            # x_client.extend(x_data)
            # plot_list.append(len(x_client))
            # y_client.extend([i_index for i in range(len(x_data))])
    # if not isRandom:
    #     plotTSNE(x_client, y_client, number_of_samples, i_index, True, False,plot_list)
    #     plotPCA(x_client, y_client, number_of_samples, i_index, True, False,plot_list)
    return  simulate_train, y1, y2

if __name__ == "__main__":
    if not os.path.exists('output/'):
        os.mkdir('output')
    if not os.path.exists('output/' + Dataset_name):
        os.mkdir('output/' + Dataset_name)
    if not os.path.exists('output/' + Dataset_name + '/' + Setup_name):
        os.mkdir('output/' + Dataset_name + '/' + Setup_name)
    if not os.path.exists(TEMP_FOLDER):
        os.mkdir(TEMP_FOLDER)

    loss_table_list = []
    meta_fed_train_kfold(Path_input, loss_table_list, number_of_samples)
    #clear_dir(TEMP_FOLDER)

    plotLosses(loss_table_list)

    # for i in range(n_folds):
    #     for k in range(number_of_samples):
    #         arr = np.load('{}fold{}_cli_{}_{}_cbt.npy'.format(Path_output, i, i, k, Setup_name))
    #         show_image(arr, i, k)
