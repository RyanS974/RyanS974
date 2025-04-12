import gradio as gr
import os
import json

# Import UI components
from ui.main_screen import create_main_screen
from ui.dataset_input import create_dataset_input, process_dataset_submission, load_example_dataset
from ui.analysis_screen import create_analysis_screen, process_analysis_request
from ui.visualization_screen import create_visualization_screen, update_visualization
from ui.classification_screen import create_classification_screen, update_classification_results
from ui.report_screen import create_report_screen, update_report, update_with_llm_analysis

# Import utility functions
from utils.llm_analyzer import run_llm_analysis
from utils.report_generator import create_report, export_report

def create_app():
    """
    Create the complete Gradio app with all tabs
    
    Returns:
        gr.Blocks: The Gradio application
    """
    with gr.Blocks(title="LLM Response Comparator", theme=gr.themes.Soft()) as app:
        # Application states to share data between tabs
        dataset_state = gr.State({})
        analysis_results_state = gr.State({})
        visualization_state = gr.State({})
        classification_results_state = gr.State({})
        report_state = gr.State({})
        
        # Create tabs
        with gr.Tabs() as tabs:
            with gr.Tab("Home", id="home_tab"):
                welcome_msg, about_info, get_started_btn = create_main_screen()
                
            with gr.Tab("Dataset Input", id="dataset_tab"):
                dataset_inputs, example_dropdown, load_example_btn, analyze_btn = create_dataset_input()
                
            with gr.Tab("Analysis", id="analysis_tab"):
                analysis_options, analysis_params, run_analysis_btn, analysis_output = create_analysis_screen()
                
            with gr.Tab("Visualization", id="viz_tab"):
                viz_options, viz_params, viz_output = create_visualization_screen()
                
            with gr.Tab("Classification", id="classification_tab"):
                classifier_options, classifier_params, run_classifier_btn, classifier_output = create_classification_screen()
                
            with gr.Tab("Report", id="report_tab"):
                report_options, generate_report_btn, llm_analysis_btn, export_btn, report_output = create_report_screen()
        
        # Set up event handlers
        
        # Main screen navigation
        get_started_btn.click(
            fn=lambda: gr.Tabs.update(selected="dataset_tab"),
            outputs=[tabs]
        )
        
        # Dataset processing
        analyze_btn.click(
            fn=process_dataset_submission,
            inputs=dataset_inputs,
            outputs=[dataset_state, gr.Tabs.update(selected="analysis_tab")]
        )
        
        # Load example dataset
        load_example_btn.click(
            fn=load_example_dataset,
            inputs=[example_dropdown],
            outputs=[dataset_inputs]
        )
        
        # Analysis
        run_analysis_btn.click(
            fn=process_analysis_request,
            inputs=[dataset_state, analysis_options, analysis_params],
            outputs=[analysis_results_state, analysis_output]
        )
        
        # Visualization updates based on analysis results
        tabs.select(
            fn=lambda tab, results: update_visualization(results, viz_options.value, viz_params.value) if tab == "viz_tab" and results else None,
            inputs=["selected", analysis_results_state],
            outputs=[viz_output]
        )
        
        viz_options.change(
            fn=update_visualization,
            inputs=[analysis_results_state, viz_options, viz_params],
            outputs=[viz_output]
        )
        
        # Classification
        run_classifier_btn.click(
            fn=update_classification_results,
            inputs=[dataset_state, classifier_options, classifier_params],
            outputs=[classification_results_state, classifier_output]
        )
        
        # Report generation
        generate_report_btn.click(
            fn=lambda results, class_results, options: update_report(create_report(results, class_results), options),
            inputs=[analysis_results_state, classification_results_state, report_options],
            outputs=[report_state, report_output]
        )
        
        # LLM meta-analysis
        llm_analysis_btn.click(
            fn=lambda report: update_with_llm_analysis(report, run_llm_analysis(report)),
            inputs=[report_state],
            outputs=[report_state, report_output]
        )
        
        # Export report
        export_btn.click(
            fn=lambda report, format: export_report(report, format),
            inputs=[report_state, gr.Dropdown(choices=["md", "html", "pdf"], value="md", label="Export Format")],
            outputs=[]
        )
        
    return app

def main():
    """
    Main function to launch the Gradio app
    """
    # Create and launch app
    app = create_app()
    app.launch(share=True)

if __name__ == "__main__":
    main()
