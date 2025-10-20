from benchmark_src.approach_interfaces.base_interface import BaseTabularEmbeddingApproach
import logging
from omegaconf import DictConfig
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

logger = logging.getLogger(__name__)

class TabuLA8BEmbedder(BaseTabularEmbeddingApproach):
    """
    TabuLA-8B embedding approach for tabular data.
    
    This approach uses the TabuLA-8B model which is specifically designed for 
    tabular data representation and can generate embeddings for table rows.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        
        # Initialize TabuLA-8B specific parameters
        self.model_name = cfg.approach.model_name if hasattr(cfg.approach, 'model_name') else "mlfoundations/tabula-8b"
        self.device = cfg.approach.device if hasattr(cfg.approach, 'device') else "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = cfg.approach.max_length if hasattr(cfg.approach, 'max_length') else 512
        self.batch_size = cfg.approach.batch_size if hasattr(cfg.approach, 'batch_size') else 8
        
        # Initialize model and tokenizer as None, will be loaded when needed
        self.model = None
        self.tokenizer = None
        
        logger.info(f"TabuLA8BEmbedder: Initialized with model {self.model_name} on device {self.device}")
    
    def load_trained_model(self):
        """
        Load the TabuLA-8B model and tokenizer.
        """
        try:
            logger.info(f"Loading TabuLA-8B model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("TabuLA-8B model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading TabuLA-8B model: {e}")
            raise
    
    def preprocessing(self, input_table: pd.DataFrame):
        """
        Preprocess the input table for TabuLA-8B.
        
        Args:
            input_table: pd.DataFrame - The table to preprocess
            
        Returns:
            list: List of preprocessed table strings
        """
        preprocessed_data = []
        
        for _, row in input_table.iterrows():
            # Convert row to string format suitable for TabuLA-8B
            table_row_string = self._convert_row_to_string(row)
            preprocessed_data.append(table_row_string)
        
        return preprocessed_data
    
    def _convert_row_to_string(self, row: pd.Series):
        """
        Convert a table row to a string format suitable for TabuLA-8B.
        
        Args:
            row: pd.Series - A single row from the table
            
        Returns:
            str: String representation of the row
        """
        # Convert all values to strings and handle NaN values
        row_dict = {}
        for col, val in row.items():
            if pd.isna(val):
                row_dict[col] = "N/A"
            else:
                row_dict[col] = str(val)
        
        # Create a structured string representation
        row_parts = []
        for col, val in row_dict.items():
            row_parts.append(f"{col}: {val}")
        
        return " | ".join(row_parts)
    
    def get_embeddings(self, texts: list):
        """
        Generate embeddings for a list of text inputs using TabuLA-8B.
        
        Args:
            texts: list - List of text strings to embed
            
        Returns:
            np.ndarray: Array of embeddings with shape [num_texts, embedding_dim]
        """
        if self.model is None or self.tokenizer is None:
            self.load_trained_model()
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize the batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use the last hidden state as embeddings
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # Average pooling
                embeddings.append(batch_embeddings.cpu().numpy())
        
        # Concatenate all batches
        return np.vstack(embeddings)
    
    def load_predictive_ml_model(self, train_df: pd.DataFrame, train_labels: pd.Series, task_type: str, dataset_information: dict):
        """
        Set up the Tabula-8B model for predictive ML tasks.
        Uses few-shot prompting with 10 randomly selected training examples.
        Args:
            train_df (pd.DataFrame): Training data.
            train_labels (pd.Series): Training labels.
            task_type (str): Either "classification" or "regression".
            dataset_information (dict): Additional dataset info.
        """
        # Load the Tabula-8B model
        self.load_trained_model()
        self.task_type = task_type
        self.unique_labels = train_labels.unique() if task_type == "classification" else None
        
        # Store training data for few-shot examples
        self.train_df = train_df.copy()
        self.train_labels = train_labels.copy()
        
        # Select random examples for few-shot prompting
        n_few_shot = getattr(self.cfg.approach, 'n_few_shot_examples', 10)
        n_examples = min(n_few_shot, len(train_df))
        random_indices = np.random.choice(len(train_df), size=n_examples, replace=False)
        self.few_shot_examples_df = train_df.iloc[random_indices].copy()
        self.few_shot_examples_labels = train_labels.iloc[random_indices].copy()
        
        logger.info(f"Tabula-8B few-shot mode enabled for {task_type} with {n_examples} examples")

    def predict_test_cases(self, test_df: pd.DataFrame, task_type: str):
        """
        Predict the target for the given test dataframe using the Tabula-8B model.
        Uses few-shot prompting with 10 training examples.
        Args:
            test_df (pd.DataFrame): The input dataframe containing test cases for prediction.
            task_type (str): Either "classification" or "regression".
        Returns:
            np.ndarray: Predictions as required by the benchmark framework.
        """
        return self._few_shot_predict(test_df, task_type)
    
    def _few_shot_predict(self, test_df: pd.DataFrame, task_type: str):
        """
        Perform few-shot prediction using Tabula-8B prompting with training examples.
        Args:
            test_df (pd.DataFrame): Test data to predict.
            task_type (str): Either "classification" or "regression".
        Returns:
            np.ndarray: Predictions as required by the benchmark framework.
        """
        predictions = []
        
        for idx, row in test_df.iterrows():
            # Create few-shot prompt with training examples
            prompt = self._create_few_shot_prompt(row, task_type)
            
            # Generate response using Tabula-8B
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.cfg.approach.max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Parse response based on task type
            if task_type == "classification":
                prediction = self._parse_classification_response(response)
            else:  # regression
                prediction = self._parse_regression_response(response)
            
            predictions.append(prediction)
        
        # Convert to numpy array and format for benchmark framework
        if task_type == "classification":
            # Return probability distribution over classes
            unique_labels = sorted(self.unique_labels)
            n_classes = len(unique_labels)
            prob_array = np.zeros((len(predictions), n_classes))
            
            for i, pred in enumerate(predictions):
                if pred in unique_labels:
                    class_idx = unique_labels.index(pred)
                    prob_array[i, class_idx] = 1.0
                else:
                    # If prediction not in training labels, assign uniform probability
                    prob_array[i, :] = 1.0 / n_classes
            
            return prob_array
        else:  # regression
            return np.array(predictions).reshape(-1, 1)
    
    def _create_few_shot_prompt(self, row: pd.Series, task_type: str):
        """
        Create a few-shot prompt with training examples and the test row.
        Args:
            row (pd.Series): Test row to predict.
            task_type (str): Either "classification" or "regression".
        Returns:
            str: Formatted prompt for Tabula-8B.
        """
        if task_type == "classification":
            prompt = "Given the following examples, predict the class for the last row:\n\n"
            
            # Add few-shot examples
            for i, (_, example_row) in enumerate(self.few_shot_examples_df.iterrows()):
                example_text = self.preprocessing(example_row)
                label = self.few_shot_examples_labels.iloc[i]
                prompt += f"Example {i+1}:\n{example_text}\nClass: {label}\n\n"
            
            # Add test row
            test_text = self.preprocessing(row)
            prompt += f"Test:\n{test_text}\nClass:"
            
        else:  # regression
            prompt = "Given the following examples, predict the value for the last row:\n\n"
            
            # Add few-shot examples
            for i, (_, example_row) in enumerate(self.few_shot_examples_df.iterrows()):
                example_text = self.preprocessing(example_row)
                label = self.few_shot_examples_labels.iloc[i]
                prompt += f"Example {i+1}:\n{example_text}\nValue: {label}\n\n"
            
            # Add test row
            test_text = self.preprocessing(row)
            prompt += f"Test:\n{test_text}\nValue:"
        
        return prompt
    
    def _zero_shot_predict(self, test_df: pd.DataFrame, task_type: str):
        """
        Perform zero-shot prediction using Tabula-8B prompting.
        Args:
            test_df (pd.DataFrame): Test data for prediction.
            task_type (str): Either "classification" or "regression".
        Returns:
            np.ndarray: Predictions in the expected format.
        """
        if self.model is None or self.tokenizer is None:
            self.load_trained_model()
        
        predictions = []
        
        for _, row in test_df.iterrows():
            # Create a zero-shot prompt
            prompt = self._create_zero_shot_prompt(row, task_type)
            
            # Tokenize and generate
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,  # Limit output length
                    do_sample=False,    # Deterministic output
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Parse the response based on task type
            if task_type == "classification":
                prediction = self._parse_classification_response(response)
            else:  # regression
                prediction = self._parse_regression_response(response)
            
            predictions.append(prediction)
        
        if task_type == "classification":
            # Convert to probability format
            unique_labels = sorted(self.unique_labels)
            probabilities = np.zeros((len(predictions), len(unique_labels)))
            for i, pred in enumerate(predictions):
                if pred in unique_labels:
                    idx = unique_labels.index(pred)
                    probabilities[i, idx] = 1.0
                else:
                    # Default to first class if prediction not recognized
                    probabilities[i, 0] = 1.0
            return probabilities
        else:
            return np.array(predictions).reshape(-1, 1)
    
    def _create_zero_shot_prompt(self, row: pd.Series, task_type: str):
        """
        Create a zero-shot prompt for the given row.
        Args:
            row (pd.Series): Single row of data.
            task_type (str): Either "classification" or "regression".
        Returns:
            str: Formatted prompt for the model.
        """
        # Convert row to text format
        row_text = self._convert_row_to_string(row)
        
        if task_type == "classification":
            unique_labels = sorted(self.unique_labels)
            labels_str = ", ".join([str(label) for label in unique_labels])
            prompt = f"""Given the following data row:
{row_text}

What is the most likely class? Choose from: {labels_str}

Answer:"""
        else:  # regression
            prompt = f"""Given the following data row:
{row_text}

What is the predicted numerical value?

Answer:"""
        
        return prompt
    
    def _parse_classification_response(self, response: str):
        """
        Parse the model's response for classification.
        Args:
            response (str): Raw model response.
        Returns:
            str: Parsed class prediction.
        """
        # Clean the response
        response = response.strip().lower()
        
        # Try to find a match with unique labels
        for label in self.unique_labels:
            if str(label).lower() in response:
                return str(label)
        
        # If no match found, return the first unique label as default
        return str(self.unique_labels[0])
    
    def _parse_regression_response(self, response: str):
        """
        Parse the model's response for regression.
        Args:
            response (str): Raw model response.
        Returns:
            float: Parsed numerical prediction.
        """
        import re
        
        # Extract numbers from the response
        numbers = re.findall(r'-?\d+\.?\d*', response)
        
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                pass
        
        # If no valid number found, return 0.0 as default
        return 0.0 
