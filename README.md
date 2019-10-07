# In diesem Workshop möchte ich euch mlflow zeigen.

Es geht darum ein Keras Model zu trainieren, welches MNIST Nummern klassifizieren soll. Später wird uns auffallen, das wir auch mit MNIST Fashion umgehen wollen. 

Wenn wir das Model trainieren, möchten wir hyperparameter tuning betreiben, um zu testen welche parameter gut sind. Natürlich möchten wir so übersichtlich wie möglich alle Trainings Hyperparameter aufzeichnen, sowie die trainierten Modelle und evtl. andere nützliche Sachen.

Nachdem wir uns für ein Model entschieden haben, möchten wir dieses deployen - und zwar mit so wenig boilerplate code wie möglich. Oft müssen vor der Modelinference noch gewisse preprocessing steps gemacht werden, um die daten in das richtige format zu bringen. 

## Vorbereitung

Wir richten ein neues conda envirnment ein

    conda create -n mlflowenv python=3.6
    source activate mlflowenv

Zu allererst brauchen wir mlflow. Das ist einfach mittles conda oder pip heruterzuladen:

    conda install mlflow

Mit einem frischen conda environment in einem leeren Ordner müssen wir für das Nutzen von mlflow erstmals ein paar Datein anlegen.

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










