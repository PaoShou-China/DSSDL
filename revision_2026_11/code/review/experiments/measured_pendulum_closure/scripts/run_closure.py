"""Original-checkpoint real-pendulum PySR discovery and human-assisted feedback."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("JULIA_NUM_THREADS", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import sympy as sp
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
NC = PROJECT / "NC实验/pendulum"
GUARDS = {}
NAMES = ["u", "theta", "q"]


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (np.ndarray,torch.Tensor)):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    return repr(value)


def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(safe(value),ensure_ascii=False,indent=2),encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def protect(path):
    path = Path(path).resolve()
    GUARDS[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


class QuietProgress:
    def __init__(self, total=None, **kwargs):
        self.n = 0
        self.description = ""
    def set_description(self, description):
        self.description = description
    def update(self, amount=1):
        self.n += amount
    def close(self):
        if self.description:
            print(self.description,flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def original_module():
    path = protect(NC / "DSSDL_mask.py")
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    cfg_node = next(n for n in tree.body if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=="config" for t in n.targets))
    cfg = ast.literal_eval(cfg_node.value) if not any(isinstance(n,ast.Call) for n in ast.walk(cfg_node.value)) else None
    env = dict(torch=torch, nn=torch.nn, Adam=torch.optim.Adam, np=np, os=os, time=time,
               DataLoader=DataLoader,TensorDataset=TensorDataset,random_split=random_split,
               tqdm=QuietProgress,f_hat=lambda x,*args,**kwargs: torch.zeros_like(x[:,:2]))
    if cfg is None:
        exec(compile(ast.Module(body=[cfg_node],type_ignores=[]),str(path),"exec"),env)
    else:
        env["config"] = cfg
    selected = {"CautiousAdam","get_adam_optimizer","load_dataset_with_one_hot","MetaNet","Trainer"}
    nodes = [n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in selected]
    assert len(nodes)==len(selected)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),env)
    return env


def source_data():
    xs,ys,records = [],[],[]
    folder = NC / "datasets/pendulum_real/train"
    for task,path in enumerate(sorted((folder/"features").glob("*.txt"))):
        label = folder/"labels"/path.name
        x,y = np.loadtxt(protect(path)),np.loadtxt(protect(label))
        np.testing.assert_array_equal(x[:,0],y[:,1])
        xs.append(x); ys.append(y)
        records.append(dict(task_id=task,file=path,labels=label,rows=len(x)))
    assert len(xs)==7
    n = {kind:dict(mean=np.concatenate(values).mean(0),std=np.concatenate(values).std(0))
         for kind,values in (("input",xs),("output",ys))}
    xn = [(x-n["input"]["mean"])/n["input"]["std"] for x in xs]
    yn = [(y-n["output"]["mean"])/n["output"]["std"] for y in ys]
    dataset = dict(raw={"train":(xs,ys)},normalized={"train":(xn,yn)},
        info=dict(input_dim=2,output_dim=2,file_counts={"train":7},normalization=n,dt=.035))
    return dataset,records


def make_model(env,dataset,checkpoint,device,spec=None):
    model = env["MetaNet"](input_dim=2,output_dim=2,one_hot_dim=7,specific_param_dim=6,
        hidden_widths=[64,64],device=device,dataset_dict=dataset,use_raw_data=False).to(device)
    if spec:
        model.f_hat_func = ExpressionPrior(spec,dataset).to(device)
    state = torch.load(protect(checkpoint),map_location=device,weights_only=True)
    model.load_state_dict(state,strict=True)
    return model


def arrays_and_loaders(env,dataset,device):
    train,val = env["load_dataset_with_one_hot"](dataset,split="train",batch_size=128,
        train_ratio=.8,device=device,use_raw_data=False)
    features,labels = train.dataset.dataset.tensors
    indices = dict(train=np.asarray(train.dataset.indices),val=np.asarray(val.dataset.indices))
    raw_x,raw_y = (np.concatenate(a) for a in dataset["raw"]["train"])
    tasks = np.concatenate([np.full(len(x),i) for i,x in enumerate(dataset["raw"]["train"][0])])
    return train,val,dict(X=features,Y=labels,raw_x=raw_x,raw_y=raw_y,tasks=tasks,**indices)


def predict_source(model,arrays,dataset):
    model.eval()
    nets,priors = [],[]
    with torch.no_grad():
        for batch in arrays["X"].split(4096):
            net,prior = model(batch)
            if not torch.is_tensor(prior):
                prior = torch.zeros_like(net)+prior
            nets.append(net.cpu().numpy()); priors.append(prior.cpu().numpy())
    net,prior = np.concatenate(nets),np.concatenate(priors)
    n = dataset["info"]["normalization"]["output"]
    return dict(normalized_full=net+prior,normalized_prior=prior,
                raw_full=(net+prior)*n["std"]+n["mean"],raw_prior=prior*n["std"]+n["mean"])


def scores(model,arrays,dataset):
    pred = predict_source(model,arrays,dataset)
    labels = arrays["Y"].cpu().numpy()
    row = dict(active_slots=np.flatnonzero(model.get_mask().detach().cpu().numpy()>.5).tolist())
    for split in ("train","val"):
        ids = arrays[split]
        for part in ("full","prior"):
            error = pred["normalized_"+part][ids]-labels[ids]
            row[f"{split}_{part}_nrmse"] = float(np.sqrt(np.mean(error**2)))
            row[f"{split}_{part}_per_output_nrmse"] = np.sqrt(np.mean(error**2,axis=0)).tolist()
    return row,pred


def stage_metrics(stage,model,arrays,dataset):
    row,pred = scores(model,arrays,dataset)
    dump(ROOT/f"stages/{stage}.json",row)
    np.savez_compressed(ROOT/f"stages/{stage}.npz",**pred)
    print("STAGE",stage,json.dumps(row),flush=True)
    rows=[]
    for file in sorted((ROOT/"stages").glob("*.json")):
        rows.append(dict(stage=file.stem,**read(file)))
    pd.DataFrame(rows).to_csv(ROOT/"tables/stage_metrics.csv",index=False)
    return row,pred


def evaluate_numpy(expr,features):
    fn = sp.lambdify(sp.symbols(" ".join(NAMES)),expr,modules="numpy")
    return np.broadcast_to(np.asarray(fn(*features.T),dtype=float),(len(features),)).copy()


def search(model,arrays,dataset,round_id):
    from pysr import PySRRegressor
    out = ROOT/f"round_{round_id:02d}"
    out.mkdir(parents=True,exist_ok=True)
    active = np.flatnonzero(model.get_mask().detach().cpu().numpy()>.5)
    if len(active)!=1:
        raise RuntimeError(f"Active dimensions changed to {active}; do not silently omit coordinates")
    q = model.one_hot_to_specific_param.weight.detach().cpu().numpy().T[:,active[0]]
    features = np.c_[arrays["raw_x"],q[arrays["tasks"]]]
    prediction = predict_source(model,arrays,dataset)
    rng = np.random.RandomState(5800+round_id)
    train_ids = np.concatenate([rng.choice(ids,min(350,len(ids)),replace=False)
        for task in range(7) for ids in [arrays["train"][arrays["tasks"][arrays["train"]]==task]]])
    va = arrays["val"]
    np.savez_compressed(out/"search_data.npz",features=features,teacher=prediction["normalized_full"],
        raw_labels=arrays["raw_y"],search_train_ids=train_ids,val_ids=va,task_ids=arrays["tasks"],q=q,active_slots=active)
    selected=[]
    for output in range(2):
        folder=out/f"output{output}"
        folder.mkdir(exist_ok=True)
        target=prediction["normalized_full"][:,output]
        n=dataset["info"]["normalization"]["output"]
        if (folder/"candidates.json").exists():
            candidates=read(folder/"candidates.json")
        else:
            start=time.perf_counter()
            sr=PySRRegressor(niterations=200,populations=16,population_size=40,
                binary_operators=["+","-","*"],unary_operators=["sin","cos"],
                maxsize=25,parsimony=.001,random_state=5800+round_id*10+output,
                deterministic=True,parallelism="serial",progress=False,verbosity=0,
                timeout_in_seconds=120,output_directory=str(folder/"pysr"),delete_tempfiles=False)
            print("SEARCH",round_id,output,len(train_ids),flush=True)
            sr.fit(features[train_ids],target[train_ids],variable_names=NAMES)
            candidates=[]
            for i,entry in sr.equations_.iterrows():
                expr=sr.sympy(index=i)
                pred=evaluate_numpy(expr,features)
                raw=sp.Float(float(n["mean"][output]))+sp.Float(float(n["std"][output]))*expr
                candidates.append(dict(index=int(i),complexity=int(entry.complexity),normalized_expression=str(expr),
                    stored_expression=str(sp.factor_terms(sp.expand(raw))),
                    val_teacher_nrmse=float(np.sqrt(np.mean((pred[va]-target[va])**2))),
                    val_data_nrmse=float(np.sqrt(np.mean((pred[va]-arrays["Y"][va,output].cpu().numpy())**2)))))
            dump(folder/"candidates.json",candidates)
            dump(folder/"search_metadata.json",dict(wall_seconds=time.perf_counter()-start,
                search_parameters=sr.get_params(),active_slots=active,physical_labels=False))
        best=min(c["val_teacher_nrmse"] for c in candidates)
        acceptable=[c for c in candidates if c["val_teacher_nrmse"]<=best*1.05+.002]
        winner=min(acceptable,key=lambda c:(c["complexity"],c["val_teacher_nrmse"]))
        dump(folder/"selection.json",winner)
        pd.DataFrame([dict(**c,selected=c["index"]==winner["index"]) for c in candidates]).to_csv(folder/"candidates.csv",index=False)
        selected.append(winner)
        print("SELECTED",output,json.dumps(winner),flush=True)
    dump(out/"selected_equations.json",dict(round=round_id,active_slot=int(active[0]),equations=selected,
        physical_labels=False,decoder=False,observed_pairs_only=True,units=["alpha/100","omega/20"]))


class ExpressionPrior(torch.nn.Module):
    def __init__(self,spec,dataset):
        super().__init__()
        self.spec=spec
        self.slot=spec["active_slot"]
        self.constants=torch.nn.Parameter(torch.tensor(spec["initial_constants"],dtype=torch.float32))
        self.expressions=[sp.sympify(e) for e in spec["templates"]]
        names=NAMES+spec["constant_names"]
        symbols=sp.symbols(" ".join(names))
        assert all(e.free_symbols<=set(symbols) for e in self.expressions)
        self.functions=[sp.lambdify(symbols,e,modules=[{"sin":torch.sin,"cos":torch.cos}]) for e in self.expressions]
        n=dataset["info"]["normalization"]
        for part in ("input","output"):
            for stat in ("mean","std"):
                self.register_buffer(f"{part}_{stat}",torch.tensor(n[part][stat],dtype=torch.float32))

    def forward(self,inp,dataset_dict=None,use_raw_data=False):
        x=inp[:,:2] if use_raw_data else inp[:,:2]*self.input_std+self.input_mean
        args=[x[:,0],x[:,1],inp[:,2+self.slot]]+list(self.constants)
        values=torch.stack([torch.zeros_like(x[:,0])+fn(*args) for fn in self.functions],dim=1)
        return values if use_raw_data else (values-self.output_mean)/self.output_std

    def current_expressions(self):
        subst=dict(zip(sp.symbols(" ".join(self.spec["constant_names"])),self.constants.detach().cpu().tolist()))
        return [str(sp.factor_terms(e.subs(subst))) for e in self.expressions]


def feedback(env,model,dataset,arrays,train,val,spec):
    prior=ExpressionPrior(spec,dataset).to(model.device)
    model.f_hat_func=prior
    sample=arrays["X"][:32]
    with torch.no_grad():
        raw_state=sample[:,:2]*prior.input_std+prior.input_mean
        q=model.one_hot_to_specific_param(sample[:,2:])[:,prior.slot]
        check_features=np.c_[raw_state.cpu().numpy(),q.cpu().numpy()]
        inp=torch.cat([sample[:,:2],model.one_hot_to_specific_param(sample[:,2:])*model.get_mask(hard=True)],dim=1)
        got=prior(inp).cpu().numpy()*prior.output_std.cpu().numpy()+prior.output_mean.cpu().numpy()
        expressions=[sp.sympify(e) for e in prior.current_expressions()]
        expected=np.column_stack([evaluate_numpy(e,check_features) for e in expressions])
        np.testing.assert_allclose(got,expected,rtol=2e-5,atol=2e-6)
    derivative_input=inp.detach().clone().requires_grad_(True)
    derivative=prior(derivative_input)
    derivative.sum().backward()
    assert torch.isfinite(derivative_input.grad).all()
    assert torch.isfinite(prior.constants.grad).all()
    model.zero_grad(set_to_none=True)
    stage_metrics("01_simplified_before_feedback",model,arrays,dataset)
    trainer=env["Trainer"](model=model,learning_rate=.01,optimizer_type="adam")
    start=time.perf_counter()
    logs=trainer.meta_train(100,train,val,reg_rate=1.)
    folder=ROOT/"feedback"
    folder.mkdir(exist_ok=True)
    torch.save(model.state_dict(),folder/"model.pt")
    pd.DataFrame(logs).to_csv(folder/"training.csv",index=False)
    dump(folder/"equations.json",dict(stored_equations=prior.current_expressions(),
        constants=prior.constants.detach().cpu().numpy(),wall_seconds=time.perf_counter()-start,
        final_epoch=99,symbolic_reg_rate=1.,active_slots=np.flatnonzero(model.get_mask().detach().cpu().numpy()>.5),
        numeric_constants_trainable=True,physical_labels=False))
    stage_metrics("02_after_feedback",model,arrays,dataset)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["discover","feedback"])
    parser.add_argument("--round",type=int,default=1)
    parser.add_argument("--checkpoint",type=Path,default=NC/"model/meta_train.pt")
    parser.add_argument("--spec",type=Path)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    seed_all(5800)
    torch.set_num_threads(2)
    for name in ("tables","stages"):
        (ROOT/name).mkdir(parents=True,exist_ok=True)
    env=original_module()
    dataset,records=source_data()
    train,val,arrays=arrays_and_loaders(env,dataset,args.device)
    spec=read(args.spec) if args.spec else None
    attach_saved_prior=spec if args.command=="discover" and args.round>1 else None
    model=make_model(env,dataset,args.checkpoint,args.device,attach_saved_prior)
    dump(ROOT/f"manifest_{args.command}_{args.round}.json",dict(args=vars(args),source_records=records,
        normalization=dataset["info"]["normalization"],original_config=env["config"],source_checkpoint=args.checkpoint,
        target_data_accessed=False,physical_labels=False,decoder=False,protected_hashes=GUARDS))
    np.savez_compressed(ROOT/"source_arrays.npz",raw_states=arrays["raw_x"],raw_labels=arrays["raw_y"],
        task_ids=arrays["tasks"],train_indices=arrays["train"],val_indices=arrays["val"])
    if args.command=="discover":
        stage_metrics("00_original_checkpoint" if args.round==1 else "02_after_feedback",model,arrays,dataset)
        search(model,arrays,dataset,args.round)
    else:
        assert spec
        feedback(env,model,dataset,arrays,train,val,spec)
    assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in GUARDS.items())
    dump(ROOT/f"validation_{args.command}_{args.round}.json",dict(passed=True,original_inputs_unchanged=True,
        target_data_accessed=False,physical_labels=False,decoder=False,protected_hashes=GUARDS))
    print("COMPLETE",args.command,args.round,flush=True)


if __name__=="__main__":
    main()
