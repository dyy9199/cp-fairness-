import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from statsmodels.stats.multitest import multipletests

def preprocess_data(path):
    df = pd.read_csv(path)

    df = df.replace('?', np.nan)

    # Binary income
    df['income'] = df['income'].apply(lambda x: 1 if '>50K' in str(x) else 0)

    #  dropna
    df = df.dropna()
    sensitive_df = df[['race', 'gender']].copy()

    # drop colunm+ + small groups+one hot encoding
    df = df.drop(columns=['education-num','fnlwgt','relationship'], errors='ignore')
    for col in df.select_dtypes(include='object').columns:
        freq = df[col].value_counts(normalize=True)
        rare = freq[freq < 0.01].index
        df[col] = df[col].apply(lambda x: 'Other' if x in rare else x)

    X_oh = pd.get_dummies(df.drop(columns=['race', 'gender','income']), drop_first=True)
    # df = pd.get_dummies(df, drop_first=True)
    df = pd.concat([X_oh, sensitive_df,df['income']], axis=1)
    return df


def split_data(df):
    X = df.drop(columns=["income", "race", "gender"])
    y = df["income"]
    race_encoder = LabelEncoder()
    gender_encoder = LabelEncoder()
    race = race_encoder.fit_transform(df["race"])
    gender = gender_encoder.fit_transform(df["gender"])

    # drop NaNs
    mask = y.notna()
    X, y, race, gender = X[mask], y[mask], race[mask], gender[mask]

    X_temp, X_test, y_temp, y_test, race_temp, race_test, gender_temp, gender_test = train_test_split(
        X, y, race, gender, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_calib, y_train, y_calib, race_train, race_calib, gender_train, gender_calib = train_test_split(
    X_temp, y_temp, race_temp, gender_temp,
    test_size=0.25, random_state=42, stratify=y_temp
)

    return X_train, X_calib, X_test, \
        y_train, y_calib, y_test, \
        race_calib, race_test, \
        gender_calib, gender_test, \
        race_encoder, gender_encoder


class BaseModelTrainer:
    def __init__(self):
        self.model = LogisticRegression(
            solver='liblinear',  
            random_state=2024,
            max_iter=1000        
        )
           
    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class ConformalPredictor:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.qhat = None

    def calibrate(self, probs, true_labels):
        true_labels = np.array(true_labels).astype(int)  
        scores = 1 - probs[np.arange(len(true_labels)), true_labels.astype(int)]
        n = len(scores)
        q_level = np.ceil((n+1)*(1-self.alpha))/n 
        self.qhat = np.quantile(scores, q_level, method='higher')  
        return self.qhat
    
    def predict(self, probs):
        threshold = 1 - self.qhat
        prediction_sets = []
        for prob in probs:
            pred_set = [i for i, p in enumerate(prob) if p >= threshold]
            prediction_sets.append(pred_set)
        return prediction_sets
    
    def evaluate_coverage(self, prediction_sets, true_labels):
        true_labels = np.array(true_labels)
        return np.mean([
            true_labels[i] in prediction_sets[i] for i in range(len(true_labels))
        ])
    



class FairnessMetrics:
    def __init__(self, prediction_sets, true_labels, sensitive_attr):        
        self.pred_sets = prediction_sets
        self.y_true = np.array(true_labels)
        self.sensitive = np.array(sensitive_attr)
        self.groups = np.unique(self.sensitive)

    def statistical_parity(self, target_class=1):
        group_probs = {}
        for g in self.groups:
            idx = np.where(self.sensitive == g)[0]
            pred_positive = [target_class in self.pred_sets[i] for i in idx]
            group_probs[g] = round(np.mean(pred_positive), 3) if len(idx) > 0 else np.nan
        return group_probs
    def equal_opportunity(self,target_class=1):
        group_probs = {}
        for g in self.groups:
            idx = np.where((self.sensitive == g) & (self.y_true == target_class))[0]
            if len(idx) == 0:
                group_probs[g] = np.nan
                continue
            covered = [target_class in self.pred_sets[i] for i in idx]
            group_probs[g] = round(np.mean(covered),3)
        return group_probs

    def predictive_parity(self, target_class=1):
        group_probs = {}
        for g in self.groups:
            idx = np.where(self.sensitive == g)[0]
            total = 0
            correct = 0
            for i in idx:
                if target_class in self.pred_sets[i]:  # predicted as positive
                    total += 1
                    if self.y_true[i] == target_class:  # actually correct
                        correct += 1
            group_probs[g] = round(correct / total, 3) if total > 0 else np.nan
        return group_probs
    
    # 3. Predictive Equality 
    def predictive_equality(self, target_class=1):
        group_probs = {}
        for g in self.groups:
            idx = np.where((self.sensitive == g) & (self.y_true == 1 - target_class))[0]
            pred_positive = [target_class in self.pred_sets[i] for i in idx]
            group_probs[g] = round(np.mean(pred_positive), 3) if len(idx) > 0 else np.nan
        return group_probs

    # 4. Equal FNR 
    def false_negative_rate_balance(self, target_class=1):
        group_probs = {}
        for g in self.groups:
            idx = np.where((self.sensitive == g) & (self.y_true == target_class))[0]
            fnr = [target_class not in self.pred_sets[i] for i in idx]  # i.e., predict wrong
            group_probs[g] = round(np.mean(fnr), 3) if len(idx) > 0 else np.nan
        return group_probs

    def subgroup_coverage(self):
        group_coverages = {}
        for g in self.groups:
            idx = np.where(self.sensitive == g)[0]
            if len(idx) == 0:
                group_coverages[g] = np.nan
            covered = [self.y_true[i] in self.pred_sets[i] for i in idx]
            group_coverages[g] = round(np.mean(covered),3)
        return group_coverages
 
    def report_all(self):
        return pd.DataFrame({
            "Statistical Parity": self.statistical_parity(),
            "Predictive_parity": self.predictive_parity(),
            "Predictive Equality": self.predictive_equality(),
            "False negative Rate": self.false_negative_rate_balance(),
            "Equal Opportunity": self.equal_opportunity(),
            "Coverage": self.subgroup_coverage()

        })


class PermutationTester:
    """permutation test for fairness metrics"""
    def __init__(self, fairness_metrics):
        self.metrics = fairness_metrics
    
    def _permute_and_test(self, metric_func, group_a, group_b, num_permutations=1000, num_bootstrap=200,**kwargs):

        mask = np.isin(self.metrics.sensitive, [group_a, group_b])
        y_true_sub = self.metrics.y_true[mask]
        pred_sets_sub = [self.metrics.pred_sets[i] for i in np.where(mask)[0]]
        sensitive_sub = self.metrics.sensitive[mask]
        
        temp_metrics = FairnessMetrics(pred_sets_sub, y_true_sub, sensitive_sub)
 ###observed gaps and sd       
        original_metrics = metric_func(temp_metrics, **kwargs)
        obs_gap = abs(original_metrics[group_a] - original_metrics[group_b]) 
        boot_gaps = []
        for _ in range(num_bootstrap):
            idx = np.random.choice(len(y_true_sub), size=len(y_true_sub), replace=True)
            y_true_boot = y_true_sub[idx]
            pred_sets_boot = [pred_sets_sub[i] for i in idx]
            sensitive_boot = sensitive_sub[idx]
            temp_metrics_boot = FairnessMetrics(pred_sets_boot, y_true_boot, sensitive_boot)
            boot_metric = metric_func(temp_metrics_boot, **kwargs)
            if group_a in boot_metric and group_b in boot_metric:
                boot_gap = abs(boot_metric[group_a] - boot_metric[group_b])
                boot_gaps.append(boot_gap)
        obs_std = np.std(boot_gaps, ddof=1)  
        S_obs = obs_gap / obs_std            
##permuation gaps and sd
        perm_gaps = []
        original_sensitive = temp_metrics.sensitive.copy()
        np.random.seed(42)  # For reproducibility
        for _ in range(num_permutations):
            temp_metrics.sensitive = np.random.permutation(original_sensitive)
            perm_metrics = metric_func(temp_metrics, **kwargs)
            if group_a in perm_metrics and group_b in perm_metrics:
                gap = abs(perm_metrics[group_a] - perm_metrics[group_b])
                perm_gaps.append(gap)

        perm_std = np.std(perm_gaps, ddof=1)  
        S_perm = perm_gaps / perm_std         

        ###p_value
        p_value = (np.sum(S_perm >= S_obs) + 1) / (len(S_perm) + 1)  
        p_value = round(p_value, 3)

        return p_value
    
    def test_binary_groups(self, metric_func, group_a=None, group_b=None, **kwargs):
        """
        gender
        """
        if group_a is None or group_b is None:
            groups = self.metrics.groups
            assert len(groups) == 2
            group_a, group_b = groups[0], groups[1]
        return self._permute_and_test(metric_func, group_a, group_b, **kwargs)
    
    def test_against_reference(self, metric_func, reference_group=4, alpha=0.05, method='holm', **kwargs):

        other_groups = [g for g in self.metrics.groups if g != reference_group]
        keys, pvals = [], []
        for g in other_groups:
            keys.append(f"{reference_group}_vs_{g}")
            pvals.append(self._permute_and_test(metric_func, reference_group, g, **kwargs))
        
        reject, pvals_adj, _, _ = multipletests(pvals, method=method, alpha=alpha)

       
        results = {}
        for k, p_raw, p_adj, rj in zip(keys, pvals, pvals_adj, reject):
            results[k] = {
                "p_raw": round(p_raw, 6),
                f"p_{method}": round(float(p_adj), 6),
                "reject_at_alpha": bool(rj),
                "alpha": alpha,
                "method": method
            }
        return results
    

if __name__ == "__main__":
    df= preprocess_data("/Users/Yue/Desktop/adult.csv")
    X_train, X_calib, X_test, \
    y_train, y_calib, y_test, \
    race_calib, race_test, \
    gender_calib, gender_test, \
    race_enc, gender_enc = split_data(df)

    print("Race groups:", dict(enumerate(race_enc.classes_)))
    print("Gender groups:", dict(enumerate(gender_enc.classes_)))
    base_model = BaseModelTrainer()
    base_model.fit(X_train, y_train)

    calib_probs = base_model.predict_proba(X_calib) ##rf outcome of calibration set
    test_probs = base_model.predict_proba(X_test) ##rf outcome of test set
    cp = ConformalPredictor(alpha=0.1) #cp inference training

    qhat = cp.calibrate(calib_probs, y_calib)
    prediction_sets = cp.predict(test_probs) ##cp interval for test set
    coverage = cp.evaluate_coverage(prediction_sets, y_test) #coverage
    print(f"Conformal Prediction completed with qhat = {qhat:.3f}")


    race_metrics = FairnessMetrics(prediction_sets, y_test,race_test) ## calculate fairness metrics

    # print(f"Split Conformal Prediction completed with coverage = {coverage:.3f}")
    result_df= race_metrics.report_all()
    print("Fairness Evaluation Results:")
    print(result_df.to_string())  
 
    race_tester = PermutationTester(race_metrics)

    race_coverage = race_tester.test_against_reference(
        metric_func=FairnessMetrics.subgroup_coverage,
        reference_group=4,
        num_permutations=1000,
            method='holm',
            alpha=0.05
    )



    print("\nRace Coverage Differences (vs White):")
    for group, result in race_coverage.items():
        p= result["p_holm"]  # holm
        print(f"{group}: p = {p:.4f}", "*" if p < 0.05 else "")

    race_sp = race_tester.test_against_reference(
        metric_func=FairnessMetrics.statistical_parity,
        reference_group=4,
        num_permutations=1000,    method='holm',
    alpha=0.05
    )
    print("\nRace statistical parity Differences (vs White):")
    for group, result in race_sp.items():
        p= result["p_holm"]  
        print(f"{group}: p = {p:.4f}", "*" if p < 0.05 else "")
    
    race_pp = race_tester.test_against_reference(
        metric_func=FairnessMetrics.predictive_parity,
        reference_group=4,
        num_permutations=1000,    method='holm',
    alpha=0.05
    )

    print("\nRace predictive_parity Differences (vs White):")
    for group, result in race_pp.items():
        p= result["p_holm"]  
        print(f"{group}: p = {p:.4f}", "*" if p < 0.05 else "")
    
    race_pe = race_tester.test_against_reference(
        metric_func=FairnessMetrics.predictive_equality,
        reference_group=4,
        num_permutations=1000,    method='holm',
    alpha=0.05
    )

    print("\nRace predictive_equality Differences (vs White):")
    for group, result in race_pe.items():
        p= result["p_holm"]  
        print(f"{group}: p = {p:.4f}", "*" if p < 0.05 else "")

    race_fnr = race_tester.test_against_reference(
        metric_func=FairnessMetrics.false_negative_rate_balance,
        reference_group=4,
        num_permutations=1000,    method='holm',
    alpha=0.05
    )

    print("\nRace false_negative_rate_balance Differences (vs White):")
    for group, result in race_fnr.items():
        p= result["p_holm"]  
        print(f"{group}: p = {p:.4f}", "*" if p < 0.05 else "")


    race_eo = race_tester.test_against_reference(
        metric_func=FairnessMetrics.equal_opportunity,
        reference_group=4,
        num_permutations=1000,    method='holm',
    alpha=0.05
    )

    print("\nRace equal opportunity balance Differences (vs White):")
    for group, result in race_eo.items():
        p= result["p_holm"] 
        print(f"{group}: p = {p:.4f}", "*" if p < 0.05 else "")


    gender_metrics = FairnessMetrics(prediction_sets, y_test,gender_test) ## calculate fairness metrics
    gender_tester = PermutationTester(gender_metrics)

    # print(f"Split Conformal Prediction completed with coverage = {coverage:.3f}")
    result_df= gender_metrics.report_all()
    print("Fairness Evaluation Results:")
    print(result_df.to_string())  # 


    gender_tester = PermutationTester(gender_metrics)
    print("\nGender Coverage Differences:")
    gender_coverage = gender_tester.test_binary_groups(
        metric_func=FairnessMetrics.subgroup_coverage,
        num_permutations=1000
    )
    g0, g1 = gender_metrics.groups
    print(f"{g0} vs {g1}: p = {gender_coverage:.4f}", "*" if gender_coverage < 0.05 else "")

    print("\nGender Statistical Parity Differences:")
    gender_sp = gender_tester.test_binary_groups(
        metric_func=FairnessMetrics.statistical_parity,
        target_class=1,
        num_permutations=1000
    )
    print(f"{g0} vs {g1}: p = {gender_sp:.4f}", "*" if gender_sp < 0.05 else "")

    print("\nGender false_negative_rate_balance Differences:")
    gender_fnr = gender_tester.test_binary_groups(
        metric_func=FairnessMetrics.false_negative_rate_balance,
        target_class=1,
        num_permutations=1000
    )
    print(f"{g0} vs {g1}: p = {gender_fnr:.4f}", "*" if gender_fnr < 0.05 else "")


    print("\nGender equal opportunity Differences:")
    gender_eo = gender_tester.test_binary_groups(
        metric_func=FairnessMetrics.equal_opportunity,
        target_class=1,
        num_permutations=1000
    )
    print(f"{g0} vs {g1}: p = {gender_eo:.4f}", "*" if gender_eo < 0.05 else "")


    print("\nGender Predictive Equality Differences:")
    gender_pe = gender_tester.test_binary_groups(
        metric_func=FairnessMetrics.predictive_equality,
        target_class=0,
        num_permutations=1000
    )
    print(f"{g0} vs {g1}: p = {gender_pe:.4f}", "*" if gender_pe < 0.05 else "")

    print("\nGender Predictive Parity Differences:")
    gender_pp = gender_tester.test_binary_groups(
        metric_func=FairnessMetrics.predictive_parity,
        target_class=1,
        num_permutations=1000
    )
    print(f"{g0} vs {g1}: p = {gender_pp:.4f}", "*" if gender_pp < 0.05 else "")
    y_test = np.array(y_test)

