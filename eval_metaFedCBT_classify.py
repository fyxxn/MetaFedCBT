from config import *
from helper import *
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from meta_model import MetaDGN
def vectorize(m):
    new_m = m.copy()
    return new_m[np.triu_indices(new_m.shape[0], k=1)].flatten()
for i in range(n_folds):
    print("Fold ",i)
    print("---------")
    for k in range(number_of_samples):
        asd_cbt_train = np.load('{}fold{}_{}_cli_{}_{}_cbt.npy'.format('./outputs/' + "ASD-LH" + '/'  , i,0, i, 0, Setup_name))
        nc_cbt_train = np.load('{}fold{}_{}_cli_{}_{}_cbt.npy'.format('./outputs/' + "NC-LH" + '/'  , i, 0, i, 0, Setup_name))
        #show_image(arr, i, k)
        nc_train_feats = vectorize(nc_cbt_train)
        asd_train_feats = vectorize(asd_cbt_train)
        nc = 0
        asd = 1

        svc = SVC()
        svc.fit([nc_train_feats, asd_train_feats], [nc, asd])

        nc_all_test_data = np.load('{}fold{}/fold{}_test_data.npy'.format('inputs/' + "NC_LH" + '/', i, i))
        asd_all_test_data = np.load('{}fold{}/fold{}_test_data.npy'.format('inputs/' + "ASD_LH" + '/', i, i))

        nc_DGN=MetaDGN(MODEL_PARAMS).to(device)
        asd_DGN=MetaDGN(MODEL_PARAMS).to(device)
        nc_DGN.eval()
        nc_DGN.load_state_dict(torch.load('{}fold{}_cli_{}_{}_{}.model'.format('./outputs/' + "NC-LH"+ '/', i, 0,i, 0, Setup_name)))
        asd_DGN.eval()
        asd_DGN.load_state_dict(torch.load('{}fold{}_cli_{}_{}_{}.model'.format('./outputs/' + "ASD-LH" + '/', i, 0, i, 0, Setup_name)))

        nc_test_cbts= meta_generate_subject_biased_cbts(nc_DGN, [d.to(device) for d in cast_data(nc_all_test_data)]).transpose(2,0,1)
        asd_test_cbts = meta_generate_subject_biased_cbts(asd_DGN, [d.to(device) for d in cast_data(asd_all_test_data)]).transpose(2,0,1)

        nc_test_feats = np.array([vectorize(nc_test_cbt) for nc_test_cbt in nc_test_cbts])
        asd_test_feats = np.array([vectorize(asd_test_cbt) for asd_test_cbt in asd_test_cbts])

        test_feats = np.concatenate([nc_test_feats, asd_test_feats], axis=0)
        test_labels = np.concatenate([np.full(nc_test_feats.shape[0], nc), np.full(asd_test_feats.shape[0], asd)] )

        preds = svc.predict(test_feats)

        acc = accuracy_score(test_labels, preds)
        prec = precision_score(test_labels, preds)
        rec = recall_score(test_labels, preds)
        f1 = f1_score(test_labels, preds)
        print("CBT Oneshot results of Hospital",k)
        print(
            f"{np.count_nonzero((preds - test_labels) == 0)} / {test_labels.shape[0]} samples correctly classified"
        )
        print(f"Acc: {acc}")
        print(f"Prec: {prec}")
        print(f"Rec: {rec}")
        print(f"F1: {f1}")
        #preds = svc.predict(test_feats)