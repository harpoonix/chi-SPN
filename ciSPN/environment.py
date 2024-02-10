from pathlib import Path

# make sure that the last dataset path is the one without interventions! - see fixme below

environment = {
    "runtime": {
        "initials": "HP",
        "machines": {
            "server": {
                "username": "ml-mwillig"
            }
        }
    },
    "experiments": {
        "base": Path("../experiments/"),
    },
    "datasets": {
        "base": Path("../datasets/"),
        "JOB_train": {
            "base": Path("JOB/JobClassification/"),
            "files": [
                "JobClassification_do(B)=U(B)_N96000_train.pkl",
                "JobClassification_do(E)=U(E)_N96000_train.pkl",
                "JobClassification_do(I)=U(I)_N96000_train.pkl",
                "JobClassification_do(Sk)=U(Sk)_N96000_train.pkl",
                "JobClassification_None_N96000_train.pkl"
            ]
        },
        "JOB_test": {
            "base": Path("JOB/JobClassification/"),
            "files": [
                # "JobClassification_do(B)=U(B)_N24000_test.pkl",
                # "JobClassification_do(E)=U(E)_N24000_test.pkl",
                # "JobClassification_do(I)=U(I)_N24000_test.pkl",
                # "JobClassification_do(Sk)=U(Sk)_N24000_test.pkl",
                # "JobClassification_None_N24000_test.pkl"
                'DoubleIntervention_do(B)=U(B)_do(E)=U(E)_N24000_test.pkl',
                'DoubleIntervention_do(E)=U(E)_do(I)=U(I)_N24000_test.pkl',  
                'DoubleIntervention_do(Sk)=U(Sk)_do(B)=U(B)_N24000_test.pkl',  
                'DoubleIntervention_do(Sk)=U(Sk)_do(I)=U(I)_N24000_test.pkl',  
            ]
        },
        "STUDENT_train" : {
            "base": Path("Student/StudentData/"),
            "files" : [
                "Student_do(C)=U(C)_N96000_train.pkl",
                "Student_do(M)=U(M)_N96000_train.pkl",
                "Student_do(Q)=U(Q)_N96000_train.pkl",
                "Student_do(T)=U(T)_N96000_train.pkl",
                "Student_None_N96000_train.pkl"
            ]
        },
        "STUDENT_test" : {
            "base": Path("Student/StudentData/"),
            "files" : [
                # "Student_do(C)=U(C)_N24000_test.pkl",
                # "Student_do(M)=U(M)_N24000_test.pkl",
                # "Student_do(Q)=U(Q)_N24000_test.pkl",
                # "Student_do(T)=U(T)_N24000_test.pkl",
                # "Student_None_N24000_test.pkl"
                'DoubleIntervention_do(C)=U(C)_do(M)=U(M)_N24000_test.pkl',
                'DoubleIntervention_do(C)=U(C)_do(Q)=U(Q)_N24000_test.pkl',
                'DoubleIntervention_do(C)=U(C)_do(T)=U(T)_N24000_test.pkl',
                'DoubleIntervention_do(T)=U(T)_do(M)=U(M)_N24000_test.pkl',
                'DoubleIntervention_do(T)=U(T)_do(Q)=U(Q)_N24000_test.pkl',
                'DoubleIntervention_do(M)=U(M)_do(Q)=U(Q)_N24000_test.pkl'
            ]
        },
        "CHC_train": {
            "base": Path("causalHealthClassification/"),
            "files": [
                "causalHealthClassification_do(A)=U(A)_N80000_train.pkl",
                "causalHealthClassification_do(F)=U(F)_N80000_train.pkl",
                "causalHealthClassification_do(H)=U(H)_N80000_train.pkl",
                "causalHealthClassification_do(M)=U(M)_N80000_train.pkl",
                "causalHealthClassification_None_N80000_train.pkl"
            ]
        },
        "CHC_test": {
            "base": Path("causalHealthClassification/"),
            "files": [
                "causalHealthClassification_do(A)=U(A)_N20000_test.pkl",
                "causalHealthClassification_do(F)=U(F)_N20000_test.pkl",
                "causalHealthClassification_do(H)=U(H)_N20000_test.pkl",
                "causalHealthClassification_do(M)=U(M)_N20000_test.pkl",
                "causalHealthClassification_None_N20000_test.pkl"
            ]
        },
        "ASIA_train": {
            "base": Path("ASIA/"),
            "files": [
                "ASIA_do(A)=UBin(A)_N80000_train.pkl",
                "ASIA_do(B)=UBin(B)_N80000_train.pkl",
                "ASIA_do(E)=UBin(E)_N80000_train.pkl",
                "ASIA_do(L)=UBin(L)_N80000_train.pkl",
                "ASIA_do(T)=UBin(T)_N80000_train.pkl",
                "ASIA_None_N80000_train.pkl"
            ]
        },
        "ASIA_test": {
            "base": Path("ASIA/"),
            "files": [
                "ASIA_do(A)=UBin(A)_N20000_test.pkl",
                "ASIA_do(B)=UBin(B)_N20000_test.pkl",
                "ASIA_do(E)=UBin(E)_N20000_test.pkl",
                "ASIA_do(L)=UBin(L)_N20000_test.pkl",
                "ASIA_do(T)=UBin(T)_N20000_test.pkl",
                "ASIA_None_N20000_test.pkl"
            ]
        },
        "CANCER_train": {
            "base": Path("CANCER/"),
            "files": [
                "CANCER_do(C)=UBin(C)_N80000_train.pkl",
                "CANCER_do(S)=UBin(S)_N80000_train.pkl",
                "CANCER_None_N80000_train.pkl"
            ]
        },
        "CANCER_test": {
            "base": Path("CANCER/"),
            "files": [
                "CANCER_do(C)=UBin(C)_N20000_test.pkl",
                "CANCER_do(S)=UBin(S)_N20000_test.pkl",
                "CANCER_None_N20000_test.pkl"
            ]
        },
        "EARTHQUAKE_train": {
            "base": Path("EARTHQUAKE/"),
            "files": [
                "EARTHQUAKE_do(A)=UBin(A)_N80000_train.pkl",
                "EARTHQUAKE_do(B)=UBin(B)_N80000_train.pkl",
                "EARTHQUAKE_do(E)=UBin(E)_N80000_train.pkl",
                "EARTHQUAKE_None_N80000_train.pkl"
            ]
        },
        "EARTHQUAKE_test": {
            "base": Path("EARTHQUAKE/"),
            "files": [
                "EARTHQUAKE_do(A)=UBin(A)_N20000_test.pkl",
                "EARTHQUAKE_do(B)=UBin(B)_N20000_test.pkl",
                "EARTHQUAKE_do(E)=UBin(E)_N20000_test.pkl",
                "EARTHQUAKE_None_N20000_test.pkl"
            ]
        },
        "CHAIN_train": {
            "base": Path("CHAIN/"),
            "files": [
                "CHAIN_do(A)=UBin(A)_N80000_train.pkl",
                "CHAIN_do(C)=UBin(C)_N80000_train.pkl",
                "CHAIN_None_N80000_train.pkl"
            ]
        },
        "CHAIN_test": {
            "base": Path("CHAIN/"),
            "files": [
                "CHAIN_do(A)=UBin(A)_N20000_test.pkl",
                "CHAIN_do(C)=UBin(C)_N20000_test.pkl",
                "CHAIN_None_N20000_test.pkl"
            ]
        },
        "hiddenObject_train": {
            "base": Path("hiddenObject_10000_2000/")
        },
        "hiddenObject_test": {
            "base": Path("hiddenObject_10000_2000/")
        },
        "hiddenObjectCorrelation_train": {
            "base": Path("hiddenObject_10000_2000_noInterventions/")
        },
        "hiddenObjectCorrelation_test": {
            "base": Path("hiddenObject_10000_2000_noInterventions/")
        },
    }
}


def get_dataset_paths(name, mode, get_base=False, no_interventions=False):
    # mode = "train" or "test"
    if mode == "test" and no_interventions:
        raise RuntimeError("Testing without interventions")

    dataset_cfg = environment["datasets"][f"{name}_{mode}"]
    base = environment["datasets"]["base"] / dataset_cfg["base"]
    if get_base:
        return base

    files = dataset_cfg["files"]
    if no_interventions:
        # FIXME assumes that the last path is the interventionless one
        files = [files[-1]]
    return [base / file for file in files]
