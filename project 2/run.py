import gradio as gr
from core.data import load_and_preprocess, calculate_flood_percentage
from core.model import train_and_evaluate, plot_data, plot_confusion_matrix

# Load the CSV files
csv_files = [
    r'Data/ASSAM-BARPETA.csv',
    r'Data/ASSAM-DHEMAJI.csv',
    r'Data/ASSAM-GOALPARA.csv',
    r'Data/ASSAM-LAKHIMPUR.csv',
    r'Data/UP-BALLIA.csv',
    r'Data/UP-BASTI.csv',
    r'Data/UP-GORAKHPUR.csv'
]

# Map districts
state_districts = {}
for file in csv_files:
    parts = file.split('-')
    state = parts[-2].split('/')[-1].strip()
    district = parts[-1].split('.')[0].strip()
    if state not in state_districts:
        state_districts[state] = []
    state_districts[state].append(district)

def load_and_analyze(state, district):
    file = f"Data/{state}-{district}.csv"
    data = load_and_preprocess(file)

    model, accuracy, report, conf_matrix = train_and_evaluate(data)
    flood_percentage = calculate_flood_percentage(data)

    rainfall_plot = plot_data(data, district, flood_percentage)
    conf_matrix_plot = plot_confusion_matrix(conf_matrix, district)

    return str(accuracy), str(report), str(flood_percentage), rainfall_plot, conf_matrix_plot

with gr.Blocks() as iface:
    with gr.Tabs():
        with gr.TabItem("ASSAM"):
            district_assam = gr.Dropdown(label="Select District", choices=state_districts.get("ASSAM", []))
            output_assam_accuracy = gr.Textbox(label="Accuracy")
            output_assam_report = gr.Textbox(label="Classification Report")
            output_assam_flood_percentage = gr.Textbox(label="Flood Percentage")
            output_assam_plot = gr.Image(label="Rainfall Data")
            output_assam_conf_matrix = gr.Image(label="Confusion Matrix")

            district_assam.change(
                load_and_analyze,
                inputs=[gr.State("ASSAM"), district_assam],
                outputs=[output_assam_accuracy, output_assam_report, output_assam_flood_percentage, output_assam_plot, output_assam_conf_matrix]
            )

        with gr.TabItem("UTTAR PRADESH"):
            district_up = gr.Dropdown(label="Select District", choices=state_districts.get('UP', []))
            output_up_accuracy = gr.Textbox(label="Accuracy")
            output_up_report = gr.Textbox(label="Classification Report")
            output_up_flood_percentage = gr.Textbox(label="Flood Percentage")
            output_up_plot = gr.Image(label="Rainfall Data")
            output_up_conf_matrix = gr.Image(label="Confusion Matrix")

            district_up.change(
                load_and_analyze,
                inputs=[gr.State("UP"), district_up],
                outputs=[output_up_accuracy, output_up_report, output_up_flood_percentage, output_up_plot, output_up_conf_matrix]
            )

iface.launch()

