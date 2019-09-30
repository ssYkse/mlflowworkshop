# In diesem Workshop möchte ich euch mlflow zeigen.

Es geht darum ein Keras Model zu trainieren, welches MNIST Nummern klassifizieren soll. Später wird uns auffallen, das wir auch mit MNIST Fashion umgehen wollen. 

Wenn wir das Model trainieren, möchten wir hyperparameter tuning betreiben, um zu testen welche parameter gut sind. Natürlich möchten wir so übersichtlich wie möglich alle Trainings Hyperparameter aufzeichnen, sowie die trainierten Modelle und evtl. andere nützliche Sachen.

Nachdem wir uns für ein Model entschieden haben, möchten wir dieses deployen - und zwar mit so wenig builerplate code wie möglich. Oft müssen vor der Modelinference noch gewisse preprocessing steps gemacht werden, um die daten in das richtige format zu bringen. 

## Vorbereitung

Mit einem frischen conda environment in einem leeren Ordner müssen wir für das Nutzen von mlflow erstmals ein paar Datein anlegen.

    conda env export > env.yaml

Wenn wir später ein mlflow 'run' laufen, dann wird dieser in einem eigenen conda environment gelaufen. Hier legen wir fest welches environment das ist.

Des weiteren brauchen wir ein 

    MLProject

file, welches die einzelnen runs beschreibt. Bis  jetzt wird dort aber nur beschrieben welches conda environment benutzt werden soll, und wie dieses projekt heißt.

## Download Data

Im ersten Schritt geht es darum, eine download pipeline zu erstellen. Da mnist durch Keras schon sehr einfach runterzuladen ist, geht dies sehr schnell. Siehe dazu download.py. 
Um diesen download auszuführen nutzen wir mlflow. Dazu fügen wir ein 'entry-point' zum MLproject, namens get_data, und führen es aus:

    mlflow run --entry-point=get_data ./

Nun ist mnist heruntergeladen, aber noch nicht im richtigen format. Außerdem hat sich ein neuer Ordner 'mlruns' erstellt. Hier wird aufgezeichnet, was mlflow für runs ausgeführt hat - mehr dazu später.

## Preprocessing





