# Tarea 3

El proyecto se estructura de las siguientes carpetas:

- /.: carpeta principal, contiene los script para ejecutar el proyecto y otras carpetas de utilidad
- /figures: carpeta para guardar las figuras exportadas en formato .pdf
    - /agg_expetiments: contiene las figuras de los experimentos agregados según lo requerido en la pregunta 4
    - /experiments: contiene las figuras de experimentos individuales
    - /test_experiments: contiene las figuras de los experimentos para testear el funcionamiento de los métodos programados
- /metrics: carpeta con los archivos .csv para generar los gráficos del reporte
- /utils: carpeta para guardar algunas funciones auxiliares a usar en el proyecto

El archivo requirements.txt contiene todas las dependencias necesarias para ejecutar este proyecto. Se resaltan las siguientes librerías:

- pandas==2.0.0
- numpy==1.24.2
- matplotlib==3.7.1
- gym==0.23.1
- torch==2.0.0

Además, los experimentos fueron ejecutados sobre python==3.9.

Ejecutar este proyecto es sencillo, sólo es necesario seguir los siguientes pasos:

1. Instalar las librerías necesarias: `pip install -r requirements.txt`
2. Replicar los experimentos: `bash run.sh`

Para Windows, se los resultados pueden ser replicados mediante el comando:
`python train_agent.py --env <env> --training_iterations <training_iterations> --lr <lr> --gamma <gamma> --batch_size <btach_size> --use_baseline <use_baseline> --reward_to_go <reward_to_go> --exp_name <exp_name>`

- `env` representa el environment del experimento
- `training_iterations` representa el número de iteraciones de entrenamiento
- `lr` representa la tasa de aprendizaje del algoritmo
- `gamma` representa el factor de descuento del algoritmo
- `batch_size` representa el tamaño del batch para el entrenamiento
- `use_baseline` es un booleano que indica si habilitar o no el uso de baseline
- `reward_to_go` es un booleano que indica si habiltiar o no el uso de reward to go
- `exp_name` representa el nombre del experimento para exportar los resultados