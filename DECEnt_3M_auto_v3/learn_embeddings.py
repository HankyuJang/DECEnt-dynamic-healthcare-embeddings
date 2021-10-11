'''
Author: -
Email: -
Last Modified: Oct 2021

NOTE: Use onehot for static embeddings

DECEnt
This code trains patient embedding from three sets of interactions.

$ python -i learn_embeddings.py --network patient_DECEnt_PF_2010-01-01 --gpu 0 --epochs 1000

'''

import time

from load_data import *
import class_network as lib
from class_network import *
import argparse
import os
from tqdm import tqdm, trange#, tqdm_notebook, tnrange

def get_statistics(user_sequence_id, user2id, item2id, static_feature_sequence, dynamic_feature_sequence,
        D2id, M2id, R2id):
    num_interactions = len(user_sequence_id)
    num_users = len(user2id) 
    num_items = len(item2id) + 1 # one extra item for "none-of-these"
    num_D = len(D2id) + 1# one extra item for "none-of-these"
    num_M = len(M2id) + 1# one extra item for "none-of-these"
    num_R = len(R2id) + 1# one extra item for "none-of-these"
    num_static_features = len(static_feature_sequence[0])
    num_dynamic_features = len(dynamic_feature_sequence[0])
    return num_interactions, num_users, num_items, num_static_features, num_dynamic_features, num_D, num_M, num_R

def print_network_statistics(num_interactions, num_users, num_items, num_static_features, num_dynamic_features, num_D, num_M, num_R):
    print("{} users, {} items ({} doctor, {} medication, {} room), {} interactions".format(num_users, num_items-1, num_D-1, num_M-1, num_R-1, num_interactions))
    print("{} user static features, {} user dynamic features".format(num_static_features, num_dynamic_features))

def initialize_loss_arrays(args):
    loss_per_timestep = np.zeros((args.epochs))
    prediction_loss_per_timestep = np.zeros((args.epochs))
    user_update_loss_per_timestep = np.zeros((args.epochs))
    item_update_loss_per_timestep = np.zeros((args.epochs))
    D_loss_per_timestep = np.zeros((args.epochs))
    M_loss_per_timestep = np.zeros((args.epochs))
    R_loss_per_timestep = np.zeros((args.epochs))
    return loss_per_timestep, prediction_loss_per_timestep, user_update_loss_per_timestep, item_update_loss_per_timestep, \
            D_loss_per_timestep, M_loss_per_timestep, R_loss_per_timestep

def initialize_dictionaries_for_tbaching():
    cached_tbatches_user = {}
    cached_tbatches_item = {}
    cached_tbatches_itemtype = {}
    cached_tbatches_interactionids = {}
    cached_tbatches_static_feature = {}
    cached_tbatches_dynamic_feature = {}
    cached_tbatches_user_timediffs = {}
    cached_tbatches_item_timediffs = {}
    cached_tbatches_previous_item = {}
    return cached_tbatches_user, cached_tbatches_item, cached_tbatches_itemtype, cached_tbatches_interactionids,\
            cached_tbatches_static_feature, cached_tbatches_dynamic_feature,\
            cached_tbatches_user_timediffs, cached_tbatches_item_timediffs, cached_tbatches_previous_item

def zero_loss():
    return [0]*9

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', required=True, help='Name of the network/dataset')
    parser.add_argument('--gpu', default=0, type=int, help='ID of the gpu. Default is 0')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs')
    parser.add_argument('--embedding_dim', default=128, type=int, help='dimension of dynamic embeddings')
    parser.add_argument('--patience', default=10, type=int, help='early stopping')
    parser.add_argument('--tbatch_timespan', default=5, type=int, 
            help='timespan of the t-batch. Longer timespan requires more GPU memory, but less number of batches are created and model is updated less frequently (training is faster).')
    parser.add_argument('--laplacian', default="laplacian_DMR", help='Name of the file that contains laplacians')
    parser.add_argument('--doctor_static', default="doctor_embedding", help='Name of the file that contains doctor static embedding')
    parser.add_argument('--medication_static', default="medication_embedding", help='Name of the file that contains medication static embedding')
    parser.add_argument('--room_static', default="room_embedding", help='Name of the file that contains room static embedding')
    parser.add_argument('--num_user_static_features', default=2, type=int, help='number of static patient features in the dataset')
    args = parser.parse_args()

    args.datapath = "data/{}.csv".format(args.network)
    args.laplacian = "data/{}.npz".format(args.laplacian)
    args.doctor_static = "data/{}.npz".format(args.doctor_static)
    args.medication_static = "data/{}.npz".format(args.medication_static)
    args.room_static = "data/{}.npz".format(args.room_static)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Load data
    [user2id, user_sequence_id, user_timediffs_sequence, user_previous_itemid_sequence,
     item2id, item_sequence_id, item_timediffs_sequence, timestamp_sequence,
     static_feature_sequence, dynamic_feature_sequence, y_true,
     item2itemtype, itemtype_sequence,
     D2id, M2id, R2id] = load_network_with_label(args)

    # Load Laplacians to use for Laplacian normalizations loss
    L_D, L_M, L_R, D_index_array, M_index_array, R_index_array = load_laplacians(args)
    L_D = torch.Tensor(L_D).cuda()
    L_M = torch.Tensor(L_M).cuda()
    L_R = torch.Tensor(L_R).cuda()
    # Save mappings for later use
    save_mappings(args, user2id, item2id, item2itemtype)

    # Print the statistics of the data before model training
    num_interactions, num_users, num_items, num_static_features, num_dynamic_features, num_D, num_M, num_R = get_statistics(user_sequence_id, user2id, item2id, static_feature_sequence, dynamic_feature_sequence,
            D2id, M2id, R2id)
    print_network_statistics(num_interactions, num_users, num_items, num_static_features, num_dynamic_features, num_D, num_M, num_R)
    # Last item per entity is a dummy item.
    D_idx_for_D_embeddings = np.arange(num_D - 1)
    M_idx_for_M_embeddings = np.arange(num_M - 1)
    R_idx_for_R_embeddings = np.arange(num_R - 1)

    train_end_idx = int(num_interactions)
    tbatch_timespan = args.tbatch_timespan

    # Model initialization
    initial_user_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0)) 
    initial_item_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))
    initial_D_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))
    initial_M_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))
    initial_R_embedding = nn.Parameter(F.normalize(torch.rand(args.embedding_dim).cuda(), dim=0))

    # Load static embeddings. 
    D_embedding_static, M_embedding_static, R_embedding_static = load_static_emb(args, D2id, M2id, R2id)
    D_embedding_static = Variable(torch.Tensor(D_embedding_static).cuda())
    M_embedding_static = Variable(torch.Tensor(M_embedding_static).cuda())
    R_embedding_static = Variable(torch.Tensor(R_embedding_static).cuda())
    # If entities do not have static embeddings, use onehot instead. E.g. uncomment the following lines
    # D_embedding_static = Variable(torch.eye(num_D).cuda())
    # M_embedding_static = Variable(torch.eye(num_M).cuda())
    # R_embedding_static = Variable(torch.eye(num_R).cuda())

    model = DECENT(args, num_static_features, num_dynamic_features, num_users, num_items, D_embedding_static.shape[1], M_embedding_static.shape[1], R_embedding_static.shape[1]).cuda()
    print(model)

    MSELoss = nn.MSELoss()

    # Embedding initialization
    user_embeddings = initial_user_embedding.repeat(num_users, 1) # initialize all users to the same embedding 
    item_embeddings = initial_item_embedding.repeat(num_items, 1) # initialize all items to the same embedding
    D_embeddings = initial_D_embedding.repeat(num_D, 1) # initialize all doctors to the same embedding
    M_embeddings = initial_M_embedding.repeat(num_M, 1) # initialize all meds to the same embedding
    R_embeddings = initial_R_embedding.repeat(num_R, 1) # initialize all rooms to the same embedding

    item_embedding_static = Variable(torch.eye(num_items).cuda()) # one-hot vectors for static embeddings
    user_embedding_static = Variable(torch.eye(num_users).cuda()) # one-hot vectors for static embeddings 

    # Optimizer
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Model train
    print("\n Training the DECENT model for {} epochs".format(args.epochs))

    # variables to help using tbatch cache between epochs
    is_first_epoch = True

    [cached_tbatches_user, cached_tbatches_item, cached_tbatches_itemtype, cached_tbatches_interactionids,\
            cached_tbatches_static_feature, cached_tbatches_dynamic_feature,\
            cached_tbatches_user_timediffs, cached_tbatches_item_timediffs, cached_tbatches_previous_item] = initialize_dictionaries_for_tbaching()

    [loss_per_timestep, prediction_loss_per_timestep, user_update_loss_per_timestep, item_update_loss_per_timestep, \
            D_loss_per_timestep, M_loss_per_timestep, R_loss_per_timestep] = initialize_loss_arrays(args)

    patience = args.patience

    ################################################################################################################################################
    # Epoch
    ################################################################################################################################################
    for ep in tqdm(range(args.epochs)):
        print("Epoch {} of {}".format(ep, args.epochs))

        epoch_start_time = time.time()
        # INITIALIZE EMBEDDING TRAJECTORY STORAGE
        user_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())
        item_embeddings_timeseries = Variable(torch.Tensor(num_interactions, args.embedding_dim).cuda())

        optimizer.zero_grad()
        reinitialize_tbatches()
        total_loss, loss, total_interaction_count, prediction_loss, user_update_loss, item_update_loss, D_loss, M_loss, R_loss = zero_loss()

        tbatch_start_time = None
        tbatch_to_insert = -1
        tbatch_full = False

        ################################################################################################################################################
        # Iterate over interactions. j is the index of the interactions
        ################################################################################################################################################
        for j in tqdm(range(train_end_idx)):
            if is_first_epoch:
                userid = user_sequence_id[j]
                itemid = item_sequence_id[j]
                itemtype = itemtype_sequence[j]
                static_feature = static_feature_sequence[j]
                dynamic_feature = dynamic_feature_sequence[j]
                user_timediff = user_timediffs_sequence[j]
                item_timediff = item_timediffs_sequence[j]

                ################################################################################################################################################
                # T-batching (Step1)
                ################################################################################################################################################
                # This is step 1 for preparing T-BATCHES
                # Later, we divide interactions in each batch into the number of types of entities
                tbatch_to_insert = max(lib.tbatchid_user[userid], lib.tbatchid_item[itemid]) + 1
                lib.tbatchid_user[userid] = tbatch_to_insert
                lib.tbatchid_item[itemid] = tbatch_to_insert

                lib.current_tbatches_user[tbatch_to_insert].append(userid)
                lib.current_tbatches_item[tbatch_to_insert].append(itemid)
                lib.current_tbatches_itemtype[tbatch_to_insert].append(itemtype)
                lib.current_tbatches_static_feature[tbatch_to_insert].append(static_feature)
                lib.current_tbatches_dynamic_feature[tbatch_to_insert].append(dynamic_feature)
                lib.current_tbatches_interactionids[tbatch_to_insert].append(j)
                lib.current_tbatches_user_timediffs[tbatch_to_insert].append(user_timediff)
                lib.current_tbatches_item_timediffs[tbatch_to_insert].append(item_timediff)
                lib.current_tbatches_previous_item[tbatch_to_insert].append(user_previous_itemid_sequence[j])

            timestamp = timestamp_sequence[j]
            if tbatch_start_time is None:
                tbatch_start_time = timestamp

            # Train the batches in the tbatch_timespan
            # if timestamp - tbatch_start_time > tbatch_timespan: # using this condition, instances in the later timesteps that do not meet this condition are NOT included in the batches!
            if (timestamp - tbatch_start_time > tbatch_timespan) or (j == train_end_idx-1): # Check if j is the last index
                # idx0: D, idx1: M, idx2: R, idx3: D, idx4: M, idx5: R, ...
                # Split each batch into three batches, based on the itemtype
                if is_first_epoch:
                    ################################################################################################################################################
                    # T-batching (Step2)
                    ################################################################################################################################################
                    tbatch_id = 0 # This is the actual tbatch_id.
                    # max_tbatch_to_insert = tbatch_to_insert # few batches are missed if we simply do this!
                    max_tbatch_to_insert = max(lib.tbatchid_user.values())
                    # iterate over each current_tbatch
                    for tbatch_to_insert in range(max_tbatch_to_insert+1): # max batch_id is inclusive.
                        # iterate over each item in the batch
                        for idx_of_interaction, itemtype_of_interaction in enumerate(lib.current_tbatches_itemtype[tbatch_to_insert]):
                            userid = lib.current_tbatches_user[tbatch_to_insert][idx_of_interaction]
                            itemid = lib.current_tbatches_item[tbatch_to_insert][idx_of_interaction]
                            static_feature = lib.current_tbatches_static_feature[tbatch_to_insert][idx_of_interaction]
                            dynamic_feature = lib.current_tbatches_dynamic_feature[tbatch_to_insert][idx_of_interaction]

                            j_interaction_id = lib.current_tbatches_interactionids[tbatch_to_insert][idx_of_interaction]
                            user_timediff = lib.current_tbatches_user_timediffs[tbatch_to_insert][idx_of_interaction]
                            item_timediff = lib.current_tbatches_item_timediffs[tbatch_to_insert][idx_of_interaction]
                            previous_item = lib.current_tbatches_previous_item[tbatch_to_insert][idx_of_interaction]

                            lib.DECEnt_tbatches_itemtype[tbatch_id] = 'D'
                            lib.DECEnt_tbatches_itemtype[tbatch_id+1] = 'M'
                            lib.DECEnt_tbatches_itemtype[tbatch_id+2] = 'R'

                            if itemtype_of_interaction=='D':
                                lib.DECEnt_tbatches_user[tbatch_id].append(userid)
                                lib.DECEnt_tbatches_item[tbatch_id].append(itemid)
                                lib.DECEnt_tbatches_static_feature[tbatch_id].append(static_feature)
                                lib.DECEnt_tbatches_dynamic_feature[tbatch_id].append(dynamic_feature)
                                lib.DECEnt_tbatches_interactionids[tbatch_id].append(j_interaction_id)
                                lib.DECEnt_tbatches_user_timediffs[tbatch_id].append(user_timediff)
                                lib.DECEnt_tbatches_item_timediffs[tbatch_id].append(item_timediff)
                                lib.DECEnt_tbatches_previous_item[tbatch_id].append(previous_item)

                            elif itemtype_of_interaction=='M':
                                lib.DECEnt_tbatches_user[tbatch_id+1].append(userid)
                                lib.DECEnt_tbatches_item[tbatch_id+1].append(itemid)
                                lib.DECEnt_tbatches_static_feature[tbatch_id+1].append(static_feature)
                                lib.DECEnt_tbatches_dynamic_feature[tbatch_id+1].append(dynamic_feature)
                                lib.DECEnt_tbatches_interactionids[tbatch_id+1].append(j_interaction_id)
                                lib.DECEnt_tbatches_user_timediffs[tbatch_id+1].append(user_timediff)
                                lib.DECEnt_tbatches_item_timediffs[tbatch_id+1].append(item_timediff)
                                lib.DECEnt_tbatches_previous_item[tbatch_id+1].append(previous_item)

                            elif itemtype_of_interaction=='R':
                                lib.DECEnt_tbatches_user[tbatch_id+2].append(userid)
                                lib.DECEnt_tbatches_item[tbatch_id+2].append(itemid)
                                lib.DECEnt_tbatches_static_feature[tbatch_id+2].append(static_feature)
                                lib.DECEnt_tbatches_dynamic_feature[tbatch_id+2].append(dynamic_feature)
                                lib.DECEnt_tbatches_interactionids[tbatch_id+2].append(j_interaction_id)
                                lib.DECEnt_tbatches_user_timediffs[tbatch_id+2].append(user_timediff)
                                lib.DECEnt_tbatches_item_timediffs[tbatch_id+2].append(item_timediff)
                                lib.DECEnt_tbatches_previous_item[tbatch_id+2].append(previous_item)

                        tbatch_id += 3
                # Reset the start time of the next tbatch
                tbatch_start_time = timestamp

                if not is_first_epoch:
                    lib.DECEnt_tbatches_user = cached_tbatches_user[timestamp]
                    lib.DECEnt_tbatches_item = cached_tbatches_item[timestamp]
                    lib.DECEnt_tbatches_itemtype = cached_tbatches_itemtype[timestamp]
                    lib.DECEnt_tbatches_interactionids = cached_tbatches_interactionids[timestamp]
                    lib.DECEnt_tbatches_static_feature = cached_tbatches_static_feature[timestamp]
                    lib.DECEnt_tbatches_dynamic_feature = cached_tbatches_dynamic_feature[timestamp]
                    lib.DECEnt_tbatches_user_timediffs = cached_tbatches_user_timediffs[timestamp]
                    lib.DECEnt_tbatches_item_timediffs = cached_tbatches_item_timediffs[timestamp]
                    lib.DECEnt_tbatches_previous_item = cached_tbatches_previous_item[timestamp]

                # print("\n")
                # print("Number of instances processed: {}".format(j+1))
                # print("Number of instances in lib.current_tbatches_user: {}".format(sum([len(lib.current_tbatches_user[batch_id]) for batch_id in lib.current_tbatches_user])))
                # print("Number of instances in lib.DECEnt_tbatches_user: {}".format(sum([len(lib.DECEnt_tbatches_user[batch_id]) for batch_id in lib.DECEnt_tbatches_user])))
                # print("\n")

                ################################################################################################################################################
                # For the batches in the tbatch_timespan, train the model
                ################################################################################################################################################
                # index upto max batch_id + 1 to include the instances in the last batch
                with trange(max(lib.DECEnt_tbatches_user.keys())+1) as progress_bar3: 
                    # Here, i is the batch_id in teh set of batches in the current tbatch_timespan
                    for i in progress_bar3:
                        # If itemtype is 'D', 'R', there are not many interactions, so the batches correspond to these get empty early.
                        if i not in lib.DECEnt_tbatches_user:
                            continue
                        num_interaction_in_batch = len(lib.DECEnt_tbatches_interactionids[i])
                        if num_interaction_in_batch == 0:
                            continue
                        total_interaction_count += num_interaction_in_batch

                        if is_first_epoch:
                            # move the tensors to GPU
                            lib.DECEnt_tbatches_user[i] = torch.LongTensor(lib.DECEnt_tbatches_user[i]).cuda()
                            lib.DECEnt_tbatches_item[i] = torch.LongTensor(lib.DECEnt_tbatches_item[i]).cuda()
                            lib.DECEnt_tbatches_interactionids[i] = torch.LongTensor(lib.DECEnt_tbatches_interactionids[i]).cuda()
                            lib.DECEnt_tbatches_static_feature[i] = torch.Tensor(lib.DECEnt_tbatches_static_feature[i]).cuda()
                            lib.DECEnt_tbatches_dynamic_feature[i] = torch.Tensor(lib.DECEnt_tbatches_dynamic_feature[i]).cuda()
                            lib.DECEnt_tbatches_user_timediffs[i] = torch.Tensor(lib.DECEnt_tbatches_user_timediffs[i]).cuda()
                            lib.DECEnt_tbatches_item_timediffs[i] = torch.Tensor(lib.DECEnt_tbatches_item_timediffs[i]).cuda()
                            lib.DECEnt_tbatches_previous_item[i] = torch.LongTensor(lib.DECEnt_tbatches_previous_item[i]).cuda()

                        tbatch_userids = lib.DECEnt_tbatches_user[i] # Recall "lib.DECEnt_tbatches_user[i]" has unique elements
                        tbatch_itemids = lib.DECEnt_tbatches_item[i] # Recall "lib.DECEnt_tbatches_item[i]" has unique elements
                        tbatch_itemtype = lib.DECEnt_tbatches_itemtype[i] # this is one string.
                        tbatch_interactionids = lib.DECEnt_tbatches_interactionids[i]
                        static_feature_tensor = Variable(lib.DECEnt_tbatches_static_feature[i]) # Recall "lib.DECEnt_tbatches_static_feature[i]" is list of list, so "static_feature_tensor" is a 2-d tensor
                        dynamic_feature_tensor = Variable(lib.DECEnt_tbatches_dynamic_feature[i]) # Recall "lib.DECEnt_tbatches_dynamic_feature[i]" is list of list, so "dynamic_feature_tensor" is a 2-d tensor

                        user_timediffs_tensor = Variable(lib.DECEnt_tbatches_user_timediffs[i]).unsqueeze(1)
                        item_timediffs_tensor = Variable(lib.DECEnt_tbatches_item_timediffs[i]).unsqueeze(1)
                        tbatch_itemids_previous = lib.DECEnt_tbatches_previous_item[i]


                        # item_embedding_previous = item_embeddings[tbatch_itemids_previous,:]
                        
                        ###############################################################################################################

                        # Step1: project user embedding
                        user_embedding_input = user_embeddings[tbatch_userids,:]
                        user_projected_embedding = model.forward(user_embedding_input, None, user_timediffs_tensor, None, None, select='project')

                        # Use the current interaction entity (e.g., med, doc, room) as both input and the vector to compute loss from.
                        # Concatenate user embedding and item embedding
                        # Get the batch of embeddings of current item
                        if tbatch_itemtype == 'D':
                            item_embedding_static_in_batch = D_embedding_static[tbatch_itemids,:]
                            item_embedding_input = D_embeddings[tbatch_itemids,:]
                        elif tbatch_itemtype == 'M':
                            item_embedding_static_in_batch = M_embedding_static[tbatch_itemids,:]
                            item_embedding_input = M_embeddings[tbatch_itemids,:]
                        elif tbatch_itemtype == 'R':
                            item_embedding_static_in_batch = R_embedding_static[tbatch_itemids,:]
                            item_embedding_input = R_embeddings[tbatch_itemids,:]

                        user_item_embedding = torch.cat(
                                [
                                    user_projected_embedding, 
                                    item_embedding_input, 
                                    item_embedding_static_in_batch,
                                    user_embedding_static[tbatch_userids,:], 
                                    static_feature_tensor
                                ], 
                                dim=1)

                        # Step2: predict the users' current item interaction
                        predicted_item_embedding = model.predict_item_embedding(user_item_embedding, itemtype=tbatch_itemtype)

                        # Loss1: prediction loss
                        loss_temp = MSELoss(predicted_item_embedding, torch.cat([item_embedding_input, item_embedding_static_in_batch], dim=1).detach())
                        loss += loss_temp
                        prediction_loss += loss_temp

                        # Step3: update dynamic embeddings based on the interaction
                        user_embedding_output = model.forward(user_embedding_input, item_embedding_input, user_timediffs_tensor, static_feature_tensor, dynamic_feature_tensor, select='user{}_update'.format(tbatch_itemtype))
                        item_embedding_output = model.forward(user_embedding_input, item_embedding_input, item_timediffs_tensor, static_feature_tensor, dynamic_feature_tensor, select='item{}_update'.format(tbatch_itemtype))

                        # Step4: Update embedding arrays
                        # item_embeddings[tbatch_itemids,:] = item_embedding_output
                        if tbatch_itemtype == 'D':
                            D_embeddings[tbatch_itemids,:] = item_embedding_output
                        elif tbatch_itemtype == 'M':
                            M_embeddings[tbatch_itemids,:] = item_embedding_output
                        elif tbatch_itemtype == 'R':
                            R_embeddings[tbatch_itemids,:] = item_embedding_output

                        user_embeddings[tbatch_userids,:] = user_embedding_output  
                        user_embeddings_timeseries[tbatch_interactionids,:] = user_embedding_output
                        item_embeddings_timeseries[tbatch_interactionids,:] = item_embedding_output # no need to fix this.

                        # Loss2: item update loss (we don not want embeddings to change dramatically per interaction)
                        loss_temp = MSELoss(item_embedding_output, item_embedding_input.detach())
                        loss += loss_temp
                        item_update_loss += loss_temp

                        # Loss3: user update loss (we don not want embeddings to change dramatically per interaction)
                        loss_temp = MSELoss(user_embedding_output, user_embedding_input.detach())
                        loss += loss_temp
                        user_update_loss += loss_temp

                        ##############
                        # Modification: do the laplacian normalization once per epoch!
                        # Loss4-6: items in the same group (e.g. doctors with same specialty) to have similar embeddings
                        if tbatch_itemtype == 'D':
                            loss_temp = torch.sum(torch.mm(torch.mm(D_embeddings[D_idx_for_D_embeddings, :].T, L_D), D_embeddings[D_idx_for_D_embeddings, :]))
                            loss += loss_temp
                            D_loss += loss_temp
                        elif tbatch_itemtype == 'M':
                            loss_temp = torch.sum(torch.mm(torch.mm(M_embeddings[M_idx_for_M_embeddings, :].T, L_M), M_embeddings[M_idx_for_M_embeddings, :]))
                            loss += loss_temp
                            M_loss += loss_temp
                        elif tbatch_itemtype == 'R':
                            loss_temp = torch.sum(torch.mm(torch.mm(R_embeddings[R_idx_for_R_embeddings, :].T, L_R), R_embeddings[R_idx_for_R_embeddings, :]))
                            loss += loss_temp
                            R_loss += loss_temp

                # At the end of t-batch, backpropagate error
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Reset loss
                loss = 0
                # item_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
                D_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
                M_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
                R_embeddings.detach_() # Detachment is needed to prevent double propagation of gradient
                user_embeddings.detach_()
                item_embeddings_timeseries.detach_() 
                user_embeddings_timeseries.detach_()
               
                # Reinitialize tbatches
                if is_first_epoch:
                    cached_tbatches_user[timestamp] = lib.DECEnt_tbatches_user
                    cached_tbatches_item[timestamp] = lib.DECEnt_tbatches_item
                    cached_tbatches_itemtype[timestamp] = lib.DECEnt_tbatches_itemtype
                    cached_tbatches_interactionids[timestamp] = lib.DECEnt_tbatches_interactionids
                    cached_tbatches_static_feature[timestamp] = lib.DECEnt_tbatches_static_feature
                    cached_tbatches_dynamic_feature[timestamp] = lib.DECEnt_tbatches_dynamic_feature
                    cached_tbatches_user_timediffs[timestamp] = lib.DECEnt_tbatches_user_timediffs
                    cached_tbatches_item_timediffs[timestamp] = lib.DECEnt_tbatches_item_timediffs
                    cached_tbatches_previous_item[timestamp] = lib.DECEnt_tbatches_previous_item
                    
                    reinitialize_tbatches()
                    tbatch_to_insert = -1

        is_first_epoch = False # as first epoch ends here
        print("Last epoch took {} minutes".format((time.time()-epoch_start_time)/60))
        # End of the epoch
        print("\nTotal loss in this epoch = %f" % (total_loss))
        print("\nPrediction loss in this epoch = %f" % (prediction_loss))

        loss_per_timestep[ep] = total_loss
        prediction_loss_per_timestep[ep] = prediction_loss
        user_update_loss_per_timestep[ep] = user_update_loss
        item_update_loss_per_timestep[ep] = item_update_loss
        D_loss_per_timestep[ep] = D_loss
        M_loss_per_timestep[ep] = M_loss
        R_loss_per_timestep[ep] = R_loss

        # Save D, M, R embeddings in item_embeddings at exact locations
        item_embeddings[D_index_array] = D_embeddings[D_idx_for_D_embeddings]
        item_embeddings[M_index_array] = M_embeddings[M_idx_for_M_embeddings]
        item_embeddings[R_index_array] = R_embeddings[R_idx_for_R_embeddings]
        # print(item_embeddings)

        item_embeddings_dystat = torch.cat([item_embeddings, item_embedding_static], dim=1)
        user_embeddings_dystat = torch.cat([user_embeddings, user_embedding_static], dim=1)
        # Save model
        # Uncomment the following line if want to save models for each epoch
        # save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)

        # Revert to initial embeddings at the end of each epoch
        # user_embeddings = initial_user_embedding.clone()
        # item_embeddings = initial_item_embedding.clone()
        user_embeddings = initial_user_embedding.repeat(num_users, 1)
        # item_embeddings = initial_item_embedding.repeat(num_items, 1)
        D_embeddings = initial_D_embedding.repeat(num_D, 1)
        M_embeddings = initial_M_embedding.repeat(num_M, 1)
        R_embeddings = initial_R_embedding.repeat(num_R, 1)

        # user_embeddings = initial_user_embedding.repeat(num_users, 1)
        # item_embeddings = initial_item_embedding.repeat(num_items, 1)

        # Save the loss at every epoch. (not necessary for training. Monitor loss over time.
        save_loss_arrays(args, loss_per_timestep, prediction_loss_per_timestep, user_update_loss_per_timestep, item_update_loss_per_timestep, D_loss_per_timestep, M_loss_per_timestep, R_loss_per_timestep)

        if ep > patience and np.argmin(loss_per_timestep[ep-patience: ep])==0:
            print("Early stopping!")
            break

        if ep // 50 == 49:
            save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)

    # Training end.
    print("\nTraining complete. Save final model")
    save_model(model, optimizer, args, ep, user_embeddings_dystat, item_embeddings_dystat, train_end_idx, user_embeddings_timeseries, item_embeddings_timeseries)

