# Ein gespeichertes MLflow model zu serven ist einfach:

	mlflow models serve -m /mnt/shared/mlflowWorkshop/0/mlruns/0/9e966915e3974922b34894abf672efa0/artifacts/my_lr_model -p 5001

Wie wir sehen, wird direkt ein Conda neues conda erstellt, und in diesem wird auf port 5001/invocations auf unsere anfrage gewaret.

Leider braucht mlflow ein bestimmtes Format als anfrage, welches für uns zum Glück gerade einfach ist:

	curl -d '{"columns":["x"], "data":[[-1],[2]]}' -H 'Content-Type: application/json; format=pandas-split' -X POST localhost:5001/invocations

Und schon bekommen wir unsere Antwort zurück!
