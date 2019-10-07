# In diesem Workshop möchte ich euch mlflow zeigen.

Es geht darum ein Keras Model zu trainieren, welches MNIST Nummern klassifizieren soll. Später wird uns auffallen, das wir auch mit MNIST Fashion umgehen wollen. 

Wenn wir das Model trainieren, möchten wir hyperparameter tuning betreiben, um zu testen welche parameter gut sind. Natürlich möchten wir so übersichtlich wie möglich alle Trainings Hyperparameter aufzeichnen, sowie die trainierten Modelle und evtl. andere nützliche Sachen.

Nachdem wir uns für ein Model entschieden haben, möchten wir dieses deployen - und zwar mit so wenig boilerplate code wie möglich. Oft müssen vor der Modelinference noch gewisse preprocessing steps gemacht werden, um die daten in das richtige format zu bringen. 

## Vorbereitung

Wir richten ein neues conda envirnment ein

    conda create -n mlflowenv python=3.6
    source activate mlflowenv

Zu allererst brauchen wir mlflow. Das ist einfach mittles conda oder pip heruterzuladen:

    pip install mlflow

Mit einem frischen conda environment in einem leeren Ordner müssen wir für das Nutzen von mlflow erstmals ein paar Datein anlegen. Da ein solches environment schon in der git repo ist, überspringen wir diesen Schritt. 

    conda env export > env.yaml

Wenn wir später ein mlflow 'run' laufen, dann wird dieser in einem eigenen conda environment gelaufen. Hier legen wir fest welches environment das ist.

Des weiteren brauchen wir ein 

    MLProject

file, welches die einzelnen runs beschreibt. Bis  jetzt wird dort aber nur beschrieben welches conda environment benutzt werden soll, und wie dieses projekt heißt:

    name: mlflow Workshop 

    conda_env: env.yaml

## Download Data

Im ersten Schritt geht es darum, eine download pipeline zu erstellen. Da mnist durch Keras schon sehr einfach runterzuladen ist, geht dies sehr schnell. Siehe dazu download.py. 
Um diesen download auszuführen nutzen wir mlflow. Dazu fügen wir einen 'entry-point' zum MLproject: get_data;

    entry_points:
      get_data:
        command: "python download.py ./data/mnist/" 

und führen es aus:

    mlflow run ./ -e get_data

Dies war unser erster mlflow run. Siehe

    mlflow run --help
   
für mehr info. Insbesondere das './' sagt mlflow, dass in der cwd das MLProject file zu betrachten ist.

Nun ist mnist heruntergeladen, aber noch nicht im richtigen format. Außerdem hat sich ein neuer Ordner 'mlruns' erstellt. Hier wird aufgezeichnet, was mlflow für runs ausgeführt hat. Beim herumstöbern im Ordner sehen wir eine Ordnerstruktur mit Dateien, welche versuchen aufzuzeichen was passiert ist.

## Exploration

Nach dem download der Daten müssen wir gucken, was wir überhaupt haben. Der MNIST Datensatz ist sehr bekannt, daher halten wir diesen Teil kurz. In expore.py betrachen wir mean und std der Daten, und führen das script wieder mit mlflow aus:

    mlflow run ./ -e explore_data

Wesentlich ist aber das wir die Bilder noch normalisiert werden müssen.

## Preprocess

Das, was wir im vorherigen Schritt gelernt haben, wenden wir nun an. Später möchten wir, dass das preprocessing automatisch passiert, wenn wir ein Bild an unseren ML-Server schicken. Daher verpacken wir den nächsten Schritt in mehr als nur einem Script.

    mlflow run ./ -e explore_data

Hier sind in MLProject die Parameter angegeben!

TODO: make function preprocess. 

Desweiteren speichern wir die behandelten MNIST daten nicht einfach in ./data/... ab, sondern lassen mlflow sich darum kümmern. Der grund dafür ist, dass die daten parameter abhängig sind (mean/std.), und daher ein teil der pipeline sind. Falls wir mean/std ändern, darf das nicht vergessen werden! Zum abspeichern der behandelten Daten führen wir

    mlflow.log_artifacts(tempdir)

im python script aus. Hier haben wir die Python API benutzt. Mehr dazu finden wir auf https://mlflow.org/docs/latest/python_api/index.html.

Lasst uns nun die Ergebnisse betrachten. Wir schauen uns den Ordner mlruns/0/...ID.../artifacts an, und sehen, dass wir dort die Daten haben. Ausserdem haben wir noch andere Informationen bezüglich diesem run, wie z.b. die parameter mean und std.

## Training and Logging

In train.py wird ein Keras Modell trainiert, welches mit den command line argumenten angepasst werden kann - hier eigentlich nur die learning rate (lr) und die Anzahl der Epochen (epochs). Der Datenordner (input_dir) ist jedoch auch wichtig - er enthält denau die MNIST daten welche vorhin durch preprocessing erzeugt wurden.

    mlflow run ./ -e train

Lässt trainiert das Modell. Am ende von train.py werden noch drei mlflow commands aus der Python API ausgeführt:

    mlflow.log_metric('loss', score[0])
    mlflow.log_metric('accuracy', score[1])
    mlflow.keras.log_model(model, "myModel")

Wie vorhin, als die behandleten MNIST daten als Artifakte abgespeichert wurden, speichert wir hier Trainingsergebnisse und das Keras Modell. Zu finden sind diese date dann im mlruns Experimentordner unter .../metrics und .../artifacts/myModel.

Betrachten wir myModel genauer, so sehen wir das in .../myModel/data ein model.h5 gespeichert ist (das Keras Modell) und in .../myModel/conda.yaml steht, was für ein Conda environment gebraucht wird um dieses Model laufen zu lassen.

Als letztes sehen wir das MLModel file, welches alles zusammen bringt:

artifact_path: myModel
flavors:
  keras:
    data: data
    keras_module: keras
    keras_version: 2.3.0
  python_function:
    data: data
    env: conda.yaml
    loader_module: mlflow.keras
    python_version: 3.6.9
run_id: e7e2b49396c94bc5a887bd201ae0eb6c
utc_time_created: '2019-10-07 12:38:35.320207'

## Packaging the Model

Es gibt die möglichkeit das keras modell, so wie wir es gespeichert haben in einem Server zur Verfügung zu stellen. Das Problem dabei ist leider, das mlflow server nur pandas dataframs welche in einem speziellen json format gespeichert sind entgegen nehmen.
Wenn z.b. img ein 2D Numpy array ist, dann kann man dies durch

    df = pd.DataFrame(img)

    df.to_json('./example.json', orient='split')

gelingen.
Bei 3D arrays ist dies leider schwieriger. Und unser Keras modell erwartet sogar 4D input tensoren (batch_size,28,28,1)!
Daher müssen wir selber uns darum kümmern, dass das Modell richtig gefüttert wird. Vorhin haben wir mlflow.keras.log_model genutzt um das model zu speichern. Nun nutzen wir mlflow.pyfunc. Was ist PyFunc? Mlflow erlaubt es einem beliebige classen zu speichern, welche ein paar methoden haben (z.b. predict, ). 

Später ist mlflow dann dazu in der Lage, diese Modelle zu laden und als server anzubieten - aber man ist nicht gezwungen mlflow server zu nutzen, man kann auch seine eigenen schreiben.


### Schritt 1

Model Erstellen

### Schritt 2

Model Speichern

### Schritt 3

Model Serving


