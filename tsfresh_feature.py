from tsai.all import *

target_list = get_UCR_univariate_list()
for t in target_list:
    print(t, "start")
    if t not in os.listdir("./dataset"):
        os.mkdir("./dataset/" + t)
    X_train, y_train, X_test, y_test  = get_UCR_data(t, return_split=True)

    ts_features_df = get_ts_features(X_train, y_train, features='min')
    df = pd.DataFrame(ts_features_df)
    df.to_csv("./dataset/" + t + "/tsfresh_train.csv", index=False)

    ts_features_df = get_ts_features(X_test, y_test, features='min')
    df = pd.DataFrame(ts_features_df)
    df.to_csv("./dataset/" + t + "/tsfresh_test.csv", index=False)
