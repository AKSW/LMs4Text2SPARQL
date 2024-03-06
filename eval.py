import json
import rdflib

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num-epochs", type=int, default=50, help="Number of training epochs (default 50)")
parser.add_argument("--dataset", type=str, required=True, choices=["coypu", "orga", "lcquad"])
parser.add_argument("--run-id", type=str, required=False, default="", help="This will be appended to the name of the json file, useful for consecutive runs of the script")

args = parser.parse_args()

run_id = args.run_id
if run_id != "":
    run_id = f"_{run_id}"

def result_sets_are_same(first, second):
    first, second = list(first), list(second)
    for item in first:
        if item not in second:
            return False
    for item in second:
        if item not in first:
            return False

    return True

def postprocess_query(q):
    q = q.replace("?", " ?")
    return q


graph_ttl="""
PREFIX : <https://abc.def/ghi/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX vcard: <http://www.w3.org/2006/vcard/ns#>
PREFIX org: <http://www.w3.org/ns/org#>

:anne a foaf:Person ; foaf:firstName "Anne" ; foaf:surname "Miller" ;
  vcard:hasAddress [ a vcard:Home ; vcard:country-name "UK" ] .
:bob a foaf:Person ; foaf:firstName "Bob" ; foaf:surname "Tanner" ;
  vcard:hasAddress [ a vcard:Home ; vcard:country-name "US" ] .

:wonderOrg a org:Organization .
:researchDep a org:OrganizationalUnit  ; org:unitOf :wonderOrg ;
  rdfs:label "Research Department" .
:marketingDep a org:OrganizationalUnit ; org:unitOf :wonderOrg ;
  rdfs:label "Marketing Department" .

:chiefResearchOfficer a org:Role . :marketingManager a org:Role .

[ a org:Membership ; org:member :anne ; org:organization :researchDep ;
  org:role :chiefResearchOfficer ] .
[ a org:Membership ; org:member :bob  ; org:organization :marketingDep ;
  org:role :marketingManager ] .
"""

models = [
    # T5 family
    "t5-small",
    "t5-base",
    "t5-large",
    "google/flan-t5-small",
    "google/flan-t5-base",
    
    # BART family
    "facebook/bart-base",
    "facebook/bart-large",
    "facebook/mbart-large-50",
    "Babelscape/mrebel-base",
    "Babelscape/mrebel-large",

    # M2M100 family
    "facebook/m2m100_418M",
    "facebook/nllb-200-distilled-600M"
]

dataset = args.dataset

g = rdflib.Graph()
if dataset == "coypu":
    g.parse("datasets/coypu_sample.ttl")
elif dataset == "orga":
    g.parse(data=graph_ttl)

result_dict = { cp: [] for cp in models }

print(dataset)
print("Model checkpoint                 |  5 |  10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50")
print("-----------------------------------------------------------------------------------")

for checkpoint in models:
    dir_prefix = checkpoint.replace("/", "_")
    results = []
    for idx in range(args.num_epochs // 5):
        try:
            with open(f"results/{dir_prefix}_{dataset}{run_id}_{idx+1}.json", "r") as fp:
                data = json.load(fp)
        except:
            break

        correct = 0
        total = 0
        for row in data:
            total += 1

            q = postprocess_query(row["generated"])
            gold = row["query"]
            try:
                s1 = g.query(q)
                s2 = g.query(gold)
                if result_sets_are_same(s1,s2):
                    correct += 1
            except:
                pass

            row["generated"] = row["generated"].replace(" ", "")
            row["query"] = row["query"].replace(" ", "")

        naive_correct = sum([ 1 if r["generated"] == r["query"] else 0 for r in data ])
        results.append(correct)
    result_dict[checkpoint] = results
    results = " | ".join(map(str,results))
    print(checkpoint, results)

final_dict = {
        "metadata": {
            "num_epochs": args.num_epochs,
            "epochs_per_eval": 5,
            "total_datapoints": total
        },
        "results": result_dict
}
with open(f"results/total_{dataset}{run_id}.json", "w") as fp:
    json.dump(final_dict, fp, indent=4)
