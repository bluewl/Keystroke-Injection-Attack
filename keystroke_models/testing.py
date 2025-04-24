import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
from keras.models import load_model
from sklearn.metrics import roc_curve
from keystroke_injection_attack import *
import datetime

SEQ_LENGTH = 70
SEQUENCES_PER_USER = 15

def extract_sequences(df, preprocessed=False):
    """
    Extract fixed-length windows of keystroke features.
    
    If preprocessed=True, df must have columns:
      ['user_id', 'HL', 'IL', 'PL', 'RL', 'keycode_norm']
      where HL/IL/PL/RL are in seconds and keycode_norm in [0,1].
    Otherwise, computes these from raw 'press_time'/'release_time' (in ms)
    and 'keycode'.
    """
    sequences, labels = [], []
    for user_id, group in df.groupby("user_id"):
        if not preprocessed:
            group = group.sort_values("PRESS_TIME")
            # compute timing features in seconds
            group["HL"] = (group["RELEASE_TIME"] - group["PRESS_TIME"]) / 1000.0
            group["IL"] = (group["PRESS_TIME"].shift(-1) - group["RELEASE_TIME"]) / 1000.0
            group["PL"] = (group["PRESS_TIME"].diff()) / 1000.0
            group["RL"] = (group["RELEASE_TIME"].diff()) / 1000.0
            # keycode normalized
            group["KEYCODE_NORM"] = group["KEYCODE"] / 255.0
            group.fillna(0, inplace=True)
        else:
            # assume already sorted / computed
            pass
        group = group.rename(columns={"KEYCODE":"keycode_norm"})
        feats = group[["PRESS_TIME", "RELEASE_TIME","HL","IL","PL","RL","keycode_norm"]].values
        # non-overlapping windows
        for start in range(0, len(feats), SEQ_LENGTH):
            end = start + SEQ_LENGTH
            if end > len(feats):
                break
            sequences.append(feats[start:end])
            labels.append(user_id)
            # cap per user
            if labels.count(user_id) >= SEQUENCES_PER_USER:
                break

    return np.array(sequences), np.array(labels)

def compute_eer(labels, scores):
    # labels: 1=genuine, 0=impostor; lower score = more genuine
    fpr, tpr, thresholds = roc_curve(labels, -np.array(scores), pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    eer_threshold = -thresholds[idx]  
    return eer, eer_threshold
    #return (fpr[idx] + fnr[idx]) / 2

def main():
    parser = argparse.ArgumentParser(description="Test TypeNet embedding model")
    parser.add_argument("--test_csv", type=str, required=True,
                        help="Path to test CSV (raw or preprocessed keystroke data)")
    parser.add_argument("--model_path", type=str, default="typenet_embedding_model.h5",
                        help="Path to saved embedding model")
    parser.add_argument("--k", type=int, required=True,
                        help="Number of test users (e.g., 1000 or 10000)")
    parser.add_argument("--G", type=int, nargs="+", default=[1,5,10],
                        help="List of gallery sizes (enrollment sequences per user)")
    parser.add_argument("--preprocessed", dest="preprocessed", action="store_true",
                        help="Set if CSV already contains HL/IL/PL/RL/keycode_norm")
    parser.add_argument("--attack", dest="attack", action="store_true",
                        help="Injection attack")
    parser.add_argument("--no_attack", dest="attack", action="store_false",
                        help="No injection attack")
    parser.add_argument("--attack_method", type=str, default="inject_backspace_retype",
                        help="Injection attack method")
    args = parser.parse_args()
    
    print(f"Tesing with parameters {args.test_csv},{args.k},{args.G}, {args.attack_method} with processed and attack flags\
        {args.preprocessed} and {args.attack} on model {args.model_path}")
    print("Batching all user scores together for single EER evaluation")
    #print("Averaging per user EER")

    # Load test data
    df = pd.read_csv(args.test_csv)
    df = df.rename(columns={"PARTICIPANT_ID":"user_id"})
    
    
    # Get all unique user_ids
    all_user_ids = df["user_id"].unique()
    all_user_ids.sort()  

    np.random.seed(43)
    selected_user_ids = np.random.choice(all_user_ids, size=args.k, replace=False)

    # Filter dataframe to only include selected users
    df = df[df["user_id"].isin(selected_user_ids)]
    
    # Extract windows and labels
    print(args.preprocessed)
    X, y = extract_sequences(df, preprocessed=args.preprocessed)
    
    # Group sequences by user
    print(len(X))
    user_seqs = defaultdict(list)
    for seq, uid in zip(X, y):
        user_seqs[uid].append(seq)
    
    # Filter users with enough sequences
    users = [uid for uid, seqs in user_seqs.items() if len(seqs) >= SEQUENCES_PER_USER]
    users = sorted(users)[:args.k]
    
    # Load embedding model
    feature_model = load_model(args.model_path, compile=False)
    
    user_infected_seq = defaultdict(list)
    embeddings_infected = {}
    attack_msg = "WE ARE UNDER ATTACK!!! \n"
    if args.attack:
        print("*** Injection Start ***")
        print(datetime.datetime.now())
        # Apply injection attack to sequences
        for uid in users:
            for i in range(len(user_seqs[uid])):
                seq = user_seqs[uid][i]
                # CHOOSE YOUR ATTACK
                if args.attack_method == "inject_backspace_retype":
                    user_infected_seq[uid].append(inject_backspace_retype(seq))
                elif args.attack_method == "inject_mirror_shift":
                    user_infected_seq[uid].append(inject_mirror_shift(seq))
                elif args.attack_method == "inject_arrows_on_trigger":
                    user_infected_seq[uid].append(inject_arrows_on_trigger(seq))
                elif args.attack_method == "inject_arrows_random_rate":
                    user_infected_seq[uid].append(inject_arrows_random_rate(seq))
                elif args.attack_method == "inject_scroll_num_on_trigger":
                    user_infected_seq[uid].append(inject_scroll_num_on_trigger(seq))
                elif args.attack_method == "inject_scroll_num_random_rate":
                    user_infected_seq[uid].append(inject_scroll_num_random_rate(seq))
                elif args.attack_method == "inject_bs_retype_random":
                    user_infected_seq[uid].append(inject_bs_retype_random(seq))

        attack_msg += " inject_backspace_retype"
        # remove press_time and release_time
        for user in users:
            user_infected_seq[user] = [seq[:, 2:] for seq in user_infected_seq[user]]

        print("*** Injection End ***")
        print(datetime.datetime.now())
        for uid in users:
            embs = feature_model.predict(
                np.stack(user_infected_seq[uid]), batch_size=256, verbose=0
            )
            embeddings_infected[uid] = embs

    # remove press_time and release_time
    for user in users:
            user_seqs[user] = [seq[:, 2:] for seq in user_seqs[user]]

    # Compute embeddings
    embeddings = {}
    for uid in users:
        embs = feature_model.predict(
            np.stack(user_seqs[uid]), batch_size=256, verbose=0
        )
        embeddings[uid] = embs

    print(datetime.datetime.now())
    print(attack_msg) if args.attack else print("There's no attack today...")

    # Evaluate EER for each gallery size G
    for G in args.G:
        scores, labels = [], []
        for uid in users:
            gallery = embeddings[uid][:G]
            queries = embeddings_infected[uid][G:SEQUENCES_PER_USER] if args.attack else embeddings[uid][G:SEQUENCES_PER_USER]
            #skip users without enough samples
            if len(queries) == 0:
                continue
            for q in queries:
                # genuine
                genu = [np.linalg.norm(g - q) for g in gallery]
                scores.append(np.mean(genu)); labels.append(1)
            potential_indices = np.array(range(len(queries)))
            for imp in users:
                user_query = queries[np.random.choice(potential_indices,replace=True)]
                if imp == uid: continue
                imp_gal = embeddings[imp][:G]
                impd = [np.linalg.norm(g - user_query) for g in imp_gal]
                #impd = [np.linalg.norm(g - q) for g in gallery]
                scores.append(np.mean(impd)); labels.append(0)
                # for imp in users:
                #     if imp == uid: continue
                #     imp_gal = embeddings[imp][:G]
                #     impd = [np.linalg.norm(g - q) for g in imp_gal]
                #     #impd = [np.linalg.norm(g - q) for g in gallery]
                #     scores.append(np.mean(impd)); labels.append(0)
        print(len(users))
        print(f"# labels: {len(labels)}, # scores: {len(scores)}")
        print(f"Unique labels: {set(labels)}")
        eer, eer_threshold = compute_eer(labels, scores)
        print(f"k={args.k}, G={G} → EER = {eer*100:.2f}% Threshold = {eer_threshold}")

    # for G in args.G:
    #     avg_eer = 0 
    #     avg_threshold = 0 
    #     for uid in users:
    #         #print("hi")
    #         scores, labels = [], []
    #         gallery = embeddings[uid][:G]
    #         queries = embeddings_infected[uid][G:SEQUENCES_PER_USER] if args.attack else embeddings[uid][G:SEQUENCES_PER_USER]
    #         #skip users without enough samples
    #         if len(queries) == 0:
    #             continue
    #         for q in queries:
    #             # genuine
    #             genu = [np.linalg.norm(g - q) for g in gallery]
    #             scores.append(np.mean(genu)); labels.append(1)
    #             # impostors: one query vs each other user's gallery
    #         potential_indices = np.array(range(len(queries)))
    #         for imp in users:
    #             impostor_queries = embeddings_infected[imp][G:SEQUENCES_PER_USER] if args.attack else embeddings[imp][G:SEQUENCES_PER_USER]
    #             impostor_query = impostor_queries[np.random.choice(potential_indices,replace=True)]
    #             if imp == uid: continue
    #             #imp_gal = embeddings[imp][:G]
    #             impd = [np.linalg.norm(g - impostor_query) for g in gallery]
    #             #impd = [np.linalg.norm(g - q) for g in gallery]
    #             scores.append(np.mean(impd)); labels.append(0)
    #             # for imp in users:
    #             #     if imp == uid: continue
    #             #     imp_gal = embeddings[imp][:G]
    #             #     impd = [np.linalg.norm(g - q) for g in imp_gal]
    #             #     #impd = [np.linalg.norm(g - q) for g in gallery]
    #             #     scores.append(np.mean(impd)); labels.append(0)
    #         eer, eer_threshold = compute_eer(labels, scores)
    #         avg_eer += eer
    #         avg_threshold += eer_threshold
        
    #     avg_eer /= len(users)
    #     avg_threshold /= len(users)

    #     print(len(users))
    #     print(f"# labels: {len(labels)}, # scores: {len(scores)}")
    #     print(f"Unique labels: {set(labels)}")
    #     #eer, eer_threshold = compute_eer(labels, scores)
    #     print(f"k={args.k}, G={G} → AVG_EER = {avg_eer*100:.2f}% AVG_Threshold = {avg_threshold}")


if __name__ == "__main__":
    main()
