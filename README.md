# In diesem Workshop möchte ich euch mlflow zeigen.

Es geht darum ein Keras Model zu trainieren, welches MNIST Nummern klassifizieren soll. Später wird uns auffallen, dass wir auch mit MNIST Fashion umgehen wollen. 

Wenn wir das Model trainieren, möchten wir Hyperparameter tuning betreiben, um zu testen welche Parameter gut sind. Natürlich möchten wir so übersichtlich wie möglich alle trainings Hyperparameter aufzeichnen, sowie die trainierten Modelle und evtl. andere nützliche Sachen.

Nachdem wir uns für ein Model entschieden haben, möchten wir dieses deployen - und zwar mit so wenig boilerplate code wie möglich. Oft müssen vor der Modelinference noch gewisse preprocessing steps gemacht werden, um die Daten in das richtige Format zu bringen. 

## Quick Start (0)

In dem Ordner ./0 zeige ich euch zuerst eine sehr schnelle Übersicht.

	python first_example.py

führt das erste experiment aus. Anstatt zu trainieren, zählen wir nur bis zehn. Die Ergebnisse werden mittels
	
	mlflow.set_tag
	mlflow.log_param
	mlflow.log_metric
	mlflow.log_artifacts

gespeichert. Angucken können wir die Ergebnisse mit

	mlflow ui

und gehen dann im Browser auf localhost:5000.

In dem zweiten Experiment trainieren wir tatsächlich ein sklearn model:

	python second_logging.py

Wieder können wir im ui die Ergebnisse betrachten. Auch können wir die Ergebnisse im Ordner ./mlruns genauer angucken.

Öffnen wir nun 0/third_serving.md. Dort wird beschrieben, wie wir das gespeicherte model als Service zur verfügung stellen können, und mittels curl/postman nutzen.

Im letzten Schritt nutzen wir die mlflow cli um das model als Docker Image zu speicher, und mit Docker zu serven. Siehe dazu 0/fourth_docker.md.

## Vorbereitung

Wir richten ein neues conda envirnment ein

    conda create -n mlflowenv python=3.6
    source activate mlflowenv

Zu allererst brauchen wir mlflow. Das ist einfach mittels pip herunterzuladen:

    pip install mlflow

Mit einem frischen conda Environment in einem leeren Ordner müssen wir für das Nutzen von mlflow erstmals ein paar Datein anlegen. Da ein solches Environment schon in der git repo ist, überspringen wir diesen Schritt. 

    conda env export > env.yaml \\ Dies würde das conda environment anlegen, welches mlflow nutzt.

Wenn wir später ein mlflow 'run' laufen lassen, wird dieser in einem eigenen conda Environment gelaufen. Hier legen wir fest welches environment das ist.

Des weiteren brauchen wir ein 

    MLProject

File, welches die einzelnen Runs beschreibt. Bis  jetzt wird dort aber nur beschrieben welches conda Environment benutzt werden soll, und wie dieses projekt heißt:

    name: mlflow Workshop 

    conda_env: env.yaml

## Download Data (1)

Im ersten Schritt geht es darum, eine download Pipeline zu erstellen. Da MNIST durch Keras schon sehr einfach herunterzuladen ist, geht dies sehr schnell. Siehe dazu download.py. 

Um diesen download auszuführen nutzen wir mlflow's CLI. Dazu fügen wir einen 'entry-point' zum MLproject: get_data;

    entry_points:
      get_data:
        command: "python download.py ./data/mnist/" 

und führen es aus:

    mlflow run ./ --entry-point get_data

Dies war unser erster mlflow run. Siehe

    mlflow run --help
   
für mehr info. Insbesondere das './' sagt mlflow, dass in der CWD das MLProject File zu betrachten ist.

Nun ist MNIST heruntergeladen, aber noch nicht im richtigen Format. Außerdem hat sich ein neuer Ordner 'mlruns' erstellt. Hier wird aufgezeichnet, was mlflow für Runs ausgeführt hat. Beim Herumstöbern im Ordner sehen wir eine Ordnerstruktur mit Dateien, welche versuchen aufzuzeichnen was passiert ist.

## Exploration (2)

Nach dem download der Daten müssen wir gucken, was wir überhaupt haben. Der MNIST Datensatz ist sehr bekannt, daher halten wir diesen Teil kurz. In expore.py betrachten wir __mean__ und __std__ der Daten, und führen das Script wieder mit mlflow aus:

    mlflow run ./ -e explore_data

Wesentlich ist, das wir die Bilder noch normalisieren müssen.

## Preprocess (2)

Das, was wir im vorherigen Schritt gelernt haben, wenden wir nun an. Später möchten wir, dass das werden Preprocessing automatisch passiert, wenn wir ein Bild an unseren ML-Server schicken. Daher verpacken wir den nächsten Schritt in mehr als nur einem Script.

    mlflow run ./ -e preprocess

Hier sind in MLProject die Parameter angegeben!

Desweiteren speichern wir die behandelten MNIST daten nicht einfach in ./data/... ab, sondern lassen mlflow sich darum kümmern. Der grund dafür ist, dass die Daten parameterabhängig sind (mean/std.), und daher ein teil der Pipeline sind. Falls wir mean/std ändern, darf das nicht vergessen werden! Zum abspeichern der behandelten Daten führen wir

    mlflow.log_artifacts(tempdir)

im python Script aus. Hier haben wir die Python API benutzt. Mehr dazu finden wir auf https://mlflow.org/docs/latest/python_api/index.html.

Lasst uns nun die Ergebnisse betrachten. Wir schauen uns den Ordner mlruns/0/...ID.../artifacts an, und sehen, dass wir dort die Daten haben. Ausserdem haben wir noch andere Informationen bezüglich diesem run, wie z.b. die parameter mean und std.

Am besten können wir natürlich wieder mit dem mlflow ui den run anschauen:
	
	mlflow ui

__Disclaimer:__ Es gibt auf github eine diskussion, ob ein solches Vorgehen Sinnvoll ist, und was es für alternativen gibt.

## Training and Logging (3)

In train.py wird ein Keras Modell trainiert, welches mit den Command Line Argumenten angepasst werden kann - hier eigentlich nur die Learning Rate (lr) und die Anzahl der Epochen (epochs). Der Datenordner (input_dir) ist jedoch auch wichtig - er enthält genau die MNIST Daten, welche vorhin durch das Preprocessing erzeugt wurden. Die ID vom vorherigen lauf kann idealerweise direkt aus dem Terminal einkopiert werden.

    mlflow run ./ -e train -P inpu_dir=./mlruns/0/...id.../artifacts

trainiert das Modell. Am Ende von train.py werden noch drei mlflow Commands aus der Python API ausgeführt:

    mlflow.log_metric('loss', score[0])
    mlflow.log_metric('accuracy', score[1])
    mlflow.keras.log_model(model, "myModel", conda_env="./conda.yaml")


__Disclaimer:__ Man muss kein conda_env spezifizieren, mlflow kann selber eins generieren. Jedoch gab es bei mir ein bug, so dass ich eins spezifiziert habe. Vieleicht liegt das daran, dass ich etwas falsch gemacht habe, vieleicht aber auch an mlflow.

Wie vorhin, als die behandleten MNIST daten als Artefakte abgespeichert wurden, speichern wir hier Trainingsergebnisse und das Keras Modell. Zu finden sind diese Date dann im ./mlruns Experimentordner unter .../metrics und .../artifacts/myModel.

Betrachten wir myModel genauer, so sehen wir das in .../myModel/data ein model.h5 gespeichert ist (das Keras Modell) und in .../myModel/conda.yaml steht, was für ein Conda Environment gebraucht wird um dieses Model laufen zu lassen.

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

## Packaging the Model (4)

Es gibt die Möglichkeit das keras Modell, so wie wir es gespeichert haben in einem Server zur Verfügung zu stellen. Das Problem dabei ist leider, das mlflow server nur Pandas Dataframes welche in einem speziellen json Format gespeichert sind entgegen nehmen kann.
Wenn z.b. img ein 2D Numpy array ist, dann kann man dies durch

    df = pd.DataFrame(img)

    df.to_json('./example.json', orient='split')

in das richtige Format bringen.
__Disclaimer:__ Dies ist blöd, denn es zwingt uns einen Data-Preprocessing Schritt aus dem mlflow model zu nehmen, und entweder eine eigene Middleware zu schreiben, oder (schlimmer noch) im Forntend das Preprocessing durchzuführen. 

Bei 3D arrays ist es leider schwieriger den tensor in ein pandas df zu verwandeln. Und unser Keras modell erwartet sogar 4D input Tensoren (batch_size,28,28,1)!
Daher müssen wir uns selber darum kümmern, dass das Modell richtig gefüttert wird. Vorhin haben wir mlflow.keras.log_model genutzt um das Model zu speichern. Nun nutzen wir mlflow.pyfunc. Was ist PyFunc? Mlflow erlaubt es einem beliebige Klassen zu speichern, welche ein paar methoden haben (z.b. predict, ), i.e. Inheritance.

Später ist mlflow dann dazu in der Lage, diese Modelle zu laden und als Server anzubieten - aber man ist nicht gezwungen mlflow Server zu nutzen - man kann auch seine eigenen schreiben.


### Schritt 1

Betrachten wir package_model.py. Unten sehen wir, dass wir ein MyPipeModel instanzieren und auch speichern, undzwar in dem Verszeichnis ./model. __Disclaimer:__ Hier wird in der Zukunft die Model Registry sinnvoll sein!

Alternativ könnten wir es natürlich wieder als Artefakt in einem mlrun speichern. Da es aber das Ende des Prozesses ist, habe ich mich dagegen entschlossen.

In '_ _init _ _' wird zurückverfolgt was mean und std waren (dies ist sicherlich nicht die eleganteste art, jedoch funktioniert es). Außerdem wird das keras model geladen und in self.model gespeichert.

In predict wird dann das pandas-json zu numpy umgewandelt, mittels .reshape() in die richtige dimension gebracht und mittels preprocess normalisiert. Schließlich bentuzen wir das keras model, und geben results zurück. Wie das json zu einem pd.Dataframe umgewandelt wird, und wie die Ergebnisse nach return zurückgeschickt werden, interessiert uns nicht weiter. Das übernimmt mlflows server.


### Schritt 2

Nun haben wir diese Klasse, welche von mlflow verstanden wird. Jeduch muss sie noch gespeichert werden. Auch hier nutzen wir mlflow, um ein pyfunc model zu erstellen. (siehe dazu die 'save' methode von PipeModel. Um diesen run auszuführen, nutzen wir

    mlflow run ./ -e package -P model_dir=./mlruns/0/...id.train.../artifacts/myModel

Betrachten wir das resultierende model: das python_model.pkl ist einfach die unten inizierte Klasse gespeichert, und das MLmodel file

    flavors:
      python_function:
        cloudpickle_version: 1.2.2
        env: conda.yaml
        loader_module: mlflow.pyfunc.model
        python_model: python_model.pkl
        python_version: 3.6.9
    utc_time_created: '2019-10-07 16:09:52.949232'

verrät uns, wie mlflow mit diesem model umgeht. Vorhin war das python_model (python_model.pkl) durch ein module_loader (mlflow.keras) ersetzt. 

### Schritt 3

Der letzte Schritt ist der einfachste. Mit 

    mlflow models serve -m ./model

setzen wir einen server auf. Alle information sind in ./model gespeichert: Das Keras modell, die preprocessing pipeline inkl. mean und std, etc.

Schicken wir ein Post request an localhost:5000/invocations mit 

{"columns":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],"index":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27],"data":[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,72,191,96,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,72,239,255,247,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,231,255,255,255,0,0,0,72,191,96,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,72,255,255,255,151,0,0,0,231,255,247,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,72,239,255,255,255,0,0,0,0,255,255,255,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,72,239,255,255,255,151,0,0,0,72,255,255,255,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,231,255,255,255,151,0,0,0,0,231,255,255,255,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,72,255,255,255,151,0,0,0,0,0,255,255,255,151,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,231,255,255,255,0,0,0,0,0,104,255,255,255,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,72,255,255,255,151,0,0,0,0,0,231,255,255,167,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,231,255,255,255,0,0,0,0,0,0,255,255,255,8,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,223,96,0,0,0,0,0,0],[0,0,0,0,0,247,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,223,96,0,0,0,0],[0,0,0,0,0,120,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,247,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,0,120,255,255,255,151,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255,255,255,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,247,255,255,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,120,255,151,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]}


als body, so erhalten wir
    
    [
        4
    ]

als Antwort.


### Schritt 4 (Überspringen, da Fehler)

Um nun ein Docker Image zu bauen, reicht folgender Command

    mlflow models build-docker -m ./model -n "my-image"

Mit 

    docker run -p 5001:8080 "my-image"

können wir nun einen containe starten, und wie im vorherigen Schritt nutzen.

### Schritt 5

Wir nutzen Paint um ein 28x28 großes grayscale Bild einer Zahl zu malen, und Speichern dieses Bild ab. Dann convertieren wir das .png in das vorhergesehene .json Format mittles

    python png_to_pandas ./Img-Name.png

(wir brauchen dazu opencv2, pandas , mittles conda install opencv pandas)
welches den output in example.json speichert.
Diesen output können wir dann verschicken.

    curl -d "@example.json" -h 'Content-Type: application/json-numpy-split' localhost:5000/invocations

(hier habe ich wieder den lokalen mlflow server genommen (port 5000))
Die MLflow Rest API erlaubt es einem nur den Request in ein paar bestimmten Formaten zu tätigen. 
['text/csv', 'application/json', 'application/json; format=pandas=records', 'application/json; format=pands-split', 'application/json-numpy-split'] sind alle möglichkeiten. Dies ist blöd, und wird in den Foren auch diskutiert. Ich hoffe, dass es dort bald Änderungen gibt.

## Modell Optimieren (5)

Wir sind glücklich, da wir ein REST server haben, der uns innerhalb 2 schnellen commands ein .png auswerten kann. Jedoch sind die Ergebnisse alle falsch. Wir möchten nun ein paar Parameter ausprobieren, um ein gutes modell zu erstellen.

Dazu ändern wir train.py, und erlauben ein paar paramter.

Es sind also epochs, lr und nr_conv_1 parameter welche das endgültige Modell beinflussen. Auch das MLProject muss angepasst werden.

    train:
      parameters:
        batch_size: {type: int, default: 32}
        epochs: {type: int , default: 3}
        lr: {type: float, default: 0.01}
        nr_conv_1: {type: int, default: 32}
        input_dir: {type: str, default: ./mlruns/0/6b3dda95d0004fb9a297088577d7eabe/artifacts}
      command: "python train.py {batch_size} {epochs} {lr} {nr_conv_1} {input_dir}"
    
Wir starten erst ein neues experiment

    mlflow experiments create -n conv1

Mit

    mlflow run . -e train -P nr_conv_1=8 --experiment-name=conv1
    mlflow run . -e train -P nr_conv_1=16 --experiment-name=conv1
    mlflow run . -e train -P nr_conv_1=24 --experiment-name=conv1
    mlflow run . -e train -P nr_conv_1=32 --experiment-name=conv1

können wir nun ein paar trainings unter diesem experiment laufen lassen. Um die Ergebnisse zu betrachten möchten wir uns natürlich nicht durch verschiedene Ordner clicken, stattdessen nutzen wir das mlflow ui:

    mlflow ui -p 5002

Öffnen wir nun unseren browser zu localhost:5002 sehen wir auf der linken zwei experiments. Ein mal Default, wo alle bisherigen 'mlflow run ...' zu finden sind, und conv1 mit den neuen experimenten.

Anstatt alle experimente per hand zu starten, können wir auch die Python API nutzen, um ein nested-run zu machen. Das bedeutet, dass wir unter einem run mehrere runs haben. Dazu benutzen wir batch_train.py, welches ähnlich wie das shell script mehrer runs startet, jedoch als __nested__ runs.

	mlflow run ./ -e train_batch -p input_dir=./.... -P epochs=1 --experiment-name=conv1

Soweit ich weiss, gibt es momentan keine Möglichkeit ein nested run vom CLI aus auszuführen.
Im UI ist es nun möglich diesen run übersichtlicher anzuschauen.

## Bestes Modell wählen 

Wir schauen uns die Ergebnisse an, und sehen das model X am besten ist. Die run URI beginnt nun mit /1/, da dies nicht mehr das default Experiment ist. Um das model zu Packagen nutzen wir wieder

    mlflow run . -e package -P model_dir=./mlruns/1/567fa5a4b71147a494d58d1d9951bf4e/artifacts/myModel --experiment-name=conv1

und schließlich

    mlflow models serve -m ./model

Wieder mit Paint und curl können wir mit dem Modell spielen.

## MNIST Fashion (6)

Nun haben wir vom Kunden neue Daten bekommen, oder das Prototy Modell hat den Kunden überzeugt, und das Modell soll nun auf den echten Daten arbeiten - hier MNIST Fashion.

Es bietet sich wieder an, ein neues experiment zu starten

    mlflow experiments create -n fashion

Der download.py code und der MLProject run müssen leicht angepasst werden. Dann kann mit 

    mlflow run . -e get_data -P type=fashion -P outdir=data/mnist-fashion/  --experiment-name=fashion

(Auf grund der Proxy-Situation musste ich leider wieder per hand mnist-fashion herunterladen).
Betracthe mnist fashion mit 

    mlflow run . -e explore_data -P in_data=./data/mnist-fashion/ --experiment-name=fashion

Die Trainingsdaten haben mean=72.940 und std=90.021.
Diese übernehmen wir fürs preprocessing:

    mlflow run . -e preprocess -P in_data=./data/mnist-fashion/ -P mean=72.94 -P std=90.021 --experiment-name=fashion

Wieder müssten wir das modell optimieren, jedoch belassen wir es bei nur einem durchgang:

    mlflow run . -e train -P input_dir=./mlruns/2/c5757d3e88894d24b940cc42bec70613/artifacts --experiment-name=fashion

Serving funktioniert wie bekannt. Erst package, dann serve:

    mlflow run . -e package -P model_dir=./mlruns/2/0f7c143bf1ca4fe1bd51df4ff1e20029/artifacts/myModel --experiment-name=fashion

    mlflow models serve -m ./model/

Schon können wir mit Paint wieder anfangen zu zeichenen!

    python png_to_pandas ./Img-Name.png
    
    curl -d "@example.json" -H 'Content-Type: application/json-numpy-split' -X POST localhost:5000/invocations

0 - T-shirt/Top  
1 - Trouser  
2 - Pullover   
3 - Dress  
4 - Coat  
5 - Sandal  
6 - Shirt  
7 - Sneaker  
8 - Bag  
9 - Ankle boot  




