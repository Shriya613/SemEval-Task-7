import tkinter as tk
from tkinter import ttk
import pandas as pd
import ast

# Load the data
posts_df = pd.read_csv('sample_data/trial_posts.csv')
fact_checks_df = pd.read_csv('sample_data/trial_fact_checks.csv')
mappings_df = pd.read_csv('sample_data/trial_data_mapping.csv')

class FactCheckViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Fact Check Viewer")
        
        # Define fonts
        self.label_font = ('Arial', 16, 'bold')
        self.text_font = ('Arial', 16)
        
        # Make the root window expandable
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Convert text column from string to tuple, handling NaN values
        posts_df['text'] = posts_df['text'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else ('', '', []))
        fact_checks_df['claim'] = fact_checks_df['claim'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else ('', '', []))
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure main frame grid weights
        main_frame.columnconfigure(1, weight=1)
        for i in range(9):
            main_frame.rowconfigure(i, weight=1)
        
        # Create post selection
        ttk.Label(main_frame, text="Select Post:", font=self.label_font).grid(row=0, column=0, sticky=tk.W)
        self.post_var = tk.StringVar()
        self.post_combo = ttk.Combobox(main_frame, textvariable=self.post_var, font=self.text_font, width=10, state="readonly")
        self.post_combo['values'] = list(posts_df['post_id'])
        self.post_combo.grid(row=0, column=1, sticky=tk.W)
        self.post_combo.bind('<<ComboboxSelected>>', self.update_display)
        
        # Create text displays
        ttk.Label(main_frame, text="Post:", font=self.label_font).grid(row=1, column=0, sticky=tk.W)
        self.original_post = tk.Text(main_frame, height=6, width=60, wrap=tk.WORD, font=self.text_font)
        self.original_post.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Empty row for spacing
        ttk.Label(main_frame).grid(row=3, column=0)
        
        ttk.Label(main_frame, text="Post (English):", font=self.label_font).grid(row=4, column=0, sticky=tk.W)
        self.translated_post = tk.Text(main_frame, height=6, width=60, wrap=tk.WORD, font=self.text_font)
        self.translated_post.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Empty row for spacing
        ttk.Label(main_frame).grid(row=6, column=0)
        
        ttk.Label(main_frame, text="Fact Check Claims:", font=self.label_font).grid(row=7, column=0, sticky=tk.W)
        self.fact_checks_original = tk.Text(main_frame, height=6, width=60, wrap=tk.WORD, font=self.text_font)
        self.fact_checks_original.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Empty row for spacing
        ttk.Label(main_frame).grid(row=9, column=0)
        
        ttk.Label(main_frame, text="Fact Check Claims (English):", font=self.label_font).grid(row=10, column=0, sticky=tk.W)
        self.fact_checks_english = tk.Text(main_frame, height=6, width=60, wrap=tk.WORD, font=self.text_font)
        self.fact_checks_english.grid(row=11, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Update grid configuration for additional rows
        for i in range(12):  # Increased for the empty spacing rows
            main_frame.rowconfigure(i, weight=1)
        
        # Select first post by default
        if len(posts_df) > 0:
            self.post_combo.set(posts_df['post_id'].iloc[0])
            self.update_display()

    def update_display(self, event=None):
        # Clear previous content
        self.original_post.delete('1.0', tk.END)
        self.translated_post.delete('1.0', tk.END)
        self.fact_checks_original.delete('1.0', tk.END)
        self.fact_checks_english.delete('1.0', tk.END)
        
        # Get selected post
        post_id = int(self.post_var.get())
        post = posts_df[posts_df['post_id'] == post_id].iloc[0]
        
        # Display post text
        self.original_post.insert(tk.END, f"[post_id: {post_id}]\n")
        self.original_post.insert(tk.END, post['text'][0])
        self.translated_post.insert(tk.END, f"[post_id: {post_id}]\n")
        self.translated_post.insert(tk.END, post['text'][1])
        
        # Get and display fact checks
        fact_check_ids = mappings_df[mappings_df['post_id'] == post_id]['fact_check_id'].tolist()
        
        if fact_check_ids:
            for fc_id in fact_check_ids:
                fact_check = fact_checks_df[fact_checks_df['fact_check_id'] == fc_id].iloc[0]
                self.fact_checks_original.insert(tk.END, f"[fact_check_id: {fc_id}]\n")
                self.fact_checks_original.insert(tk.END, fact_check['claim'][0] + "\n\n")
                
                self.fact_checks_english.insert(tk.END, f"[fact_check_id: {fc_id}]\n")
                self.fact_checks_english.insert(tk.END, fact_check['claim'][1] + "\n\n")

def main():
    root = tk.Tk()
    # Set initial window size (width x height)
    root.geometry("800x800")  # Increased height to accommodate larger text boxes
    app = FactCheckViewer(root)
    root.mainloop()

if __name__ == "__main__":
    main()
