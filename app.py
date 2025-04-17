import tkinter as tk
from tkinter import ttk, messagebox
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import threading

# Load Pre-trained Model & Tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'distilbert-base-uncased'

# Load model and tokenizer in memory before GUI starts
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=6,
    id2label={
        0: 'sadness',
        1: 'joy',
        2: 'love',
        3: 'anger',
        4: 'fear',
        5: 'surprise'
    }
).to(device)
model.eval()

class EmotionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Recognition")
        self.root.geometry("400x300")
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure('TLabel', font=('Arial', 12))
        self.style.configure('TButton', font=('Arial', 12))

        # Create GUI components
        self.create_widgets()
        
        # Initialize prediction thread
        self.prediction_thread = None

    def create_widgets(self):
        # Input Frame
        input_frame = ttk.Frame(self.root)
        input_frame.pack(pady=10, padx=20, fill='x')

        ttk.Label(input_frame, text="Enter Text:").pack(anchor='w')
        self.text_input = ttk.Entry(input_frame, width=50)
        self.text_input.pack(pady=5)

        # Button Frame
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)

        self.predict_btn = ttk.Button(
            button_frame, 
            text="Analyze Emotion", 
            command=self.start_prediction
        )
        self.predict_btn.pack(side='left', padx=5)

        self.clear_btn = ttk.Button(
            button_frame, 
            text="Clear", 
            command=self.clear_input
        )
        self.clear_btn.pack(side='left', padx=5)

        # Result Frame
        result_frame = ttk.Frame(self.root)
        result_frame.pack(pady=10, padx=20, fill='x')

        self.result_label = ttk.Label(
            result_frame, 
            text="Emotion: ", 
            font=('Arial', 14, 'bold')
        )
        self.result_label.pack(anchor='w')

        self.confidence_label = ttk.Label(
            result_frame, 
            text="Confidence: ", 
            font=('Arial', 12)
        )
        self.confidence_label.pack(anchor='w')

        # Status Bar
        self.status_bar = ttk.Label(
            self.root, 
            text="Ready", 
            relief='sunken', 
            anchor='w'
        )
        self.status_bar.pack(side='bottom', fill='x')

    def start_prediction(self):
        text = self.text_input.get().strip()
        if not text:
            messagebox.showwarning("Input Error", "Please enter some text!")
            return

        # Disable buttons during processing
        self.toggle_buttons(False)
        self.update_status("Processing...")
        
        # Run prediction in separate thread
        self.prediction_thread = threading.Thread(
            target=self.run_prediction, 
            args=(text,)
        )
        self.prediction_thread.start()
        
        # Check thread status periodically
        self.root.after(100, self.check_thread)

    def run_prediction(self, text):
        try:
            inputs = tokenizer(
                text, 
                return_tensors='pt', 
                truncation=True, 
                padding=True, 
                max_length=128
            ).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_label = torch.argmax(probs).item()
            
            self.root.after(0, self.show_results, {
                'label': model.config.id2label[pred_label],
                'confidence': probs[0][pred_label].item()
            })
            
        except Exception as e:
            self.root.after(0, self.show_error, str(e))

    def check_thread(self):
        if self.prediction_thread.is_alive():
            self.root.after(100, self.check_thread)
        else:
            self.toggle_buttons(True)
            self.update_status("Ready")

    def show_results(self, result):
        self.result_label.config(text=f"Emotion: {result['label']}")
        self.confidence_label.config(
            text=f"Confidence: {result['confidence']:.2%}"
        )
        self.update_status("Prediction complete")

    def show_error(self, error_msg):
        messagebox.showerror("Error", f"Prediction failed: {error_msg}")
        self.clear_results()
        self.update_status("Error occurred")

    def clear_input(self):
        self.text_input.delete(0, tk.END)
        self.clear_results()

    def clear_results(self):
        self.result_label.config(text="Emotion: ")
        self.confidence_label.config(text="Confidence: ")

    def toggle_buttons(self, state):
        self.predict_btn['state'] = 'normal' if state else 'disabled'
        self.clear_btn['state'] = 'normal' if state else 'disabled'

    def update_status(self, message):
        self.status_bar.config(text=message)

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()
