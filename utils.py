import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict, Optional, Union
import warnings

class TimeSeriesMLProcessor:
    """
    A comprehensive processor for time series data with moving windows and transition detection.
    
    Features:
    - Creates moving window features from time series data
    - Generates transition/no-transition labels
    - Supports both transductive (block-based) and inductive (animal-based) splits
    - Handles per-animal and pooled analysis
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the processor with your dataframe.
        
        Args:
            df: DataFrame with columns including 'animal', 'block', 'state' and feature columns
        """
        self.df = df.copy()
        self.feature_columns = [col for col in df.columns 
                               if col not in ['animal', 'block', 'subblock', 'window_index', 
                                            'condition', 'state', 'window_start_s', 'window_end_s']]
        self.processed_data = None
        
    def create_moving_windows(self, window_size: int = 5, 
                            feature_cols: Optional[List[str]] = None,
                            aggregation_methods: List[str] = ['mean', 'std'],
                            include_raw_windows: bool = True,
                            include_aggregated: bool = True) -> pd.DataFrame:
        """
        Create moving window features from the time series data.
        
        Args:
            window_size: Size of the moving window
            feature_cols: List of feature columns to use (if None, uses all feature columns)
            aggregation_methods: Methods to aggregate within windows ['mean', 'std', 'min', 'max', 'median']
            include_raw_windows: If True, includes raw values from each timestep in the window
            include_aggregated: If True, includes aggregated statistics
            
        Returns:
            DataFrame with moving window features and transition labels
        """
        if feature_cols is None:
            feature_cols = self.feature_columns
            
        processed_rows = []
        
        for animal in self.df['animal'].unique():
            animal_data = self.df[self.df['animal'] == animal]
            
            for block in animal_data['block'].unique():
                block_data = animal_data[animal_data['block'] == block].reset_index(drop=True)
                
                # Create moving windows for this block
                for i in range(window_size - 1, len(block_data)):
                    window_data = block_data.iloc[i - window_size + 1:i + 1]
                    
                    # Create row for this window
                    row = {
                        'animal': animal,
                        'block': block,
                        'window_end_index': i,
                        'current_state': block_data.iloc[i]['state']}
                    #return summary
                    
                    # Determine transition type
                    if i > 0:
                        prev_state = block_data.iloc[i-1]['state']
                        current_state = block_data.iloc[i]['state']
                        
                        # Create detailed transition labels
                        if prev_state != current_state:
                            # Actual transition occurred
                            if prev_state.lower() in ['wake', 'w'] and current_state.lower() in ['nrem', 'n']:
                                row['transition_type'] = 'wake_to_nrem'
                                row['transition_category'] = 'wake_to_nrem'
                            elif prev_state.lower() in ['nrem', 'n'] and current_state.lower() in ['wake', 'w']:
                                row['transition_type'] = 'nrem_to_wake'
                                row['transition_category'] = 'nrem_to_wake'
                            else:
                                # Handle other potential states (REM, etc.)
                                row['transition_type'] = f'{prev_state}_to_{current_state}'
                                row['transition_category'] = 'other_transition'
                            
                            row['is_transition'] = 1
                        else:
                            # No transition - staying in same state
                            if current_state.lower() in ['wake', 'w']:
                                row['transition_type'] = 'stay_wake'
                                row['transition_category'] = 'stay_wake'
                            elif current_state.lower() in ['nrem', 'n']:
                                row['transition_type'] = 'stay_nrem'
                                row['transition_category'] = 'stay_nrem'
                            else:
                                row['transition_type'] = f'stay_{current_state}'
                                row['transition_category'] = f'stay_{current_state}'
                            
                            row['is_transition'] = 0
                    else:
                        # First point - no previous state to compare
                        current_state = block_data.iloc[i]['state']
                        row['transition_type'] = f'initial_{current_state}'
                        row['transition_category'] = f'stay_{current_state}'  # Treat as staying in current state
                        row['is_transition'] = 0
                    
                    # Add previous state for context
                    row['previous_state'] = block_data.iloc[i-1]['state'] if i > 0 else None
                    
                    # Add raw window data (each timestep as separate features)
                    if include_raw_windows:
                        for t, (_, window_row) in enumerate(window_data.iterrows()):
                            for col in feature_cols:
                                row[f'{col}_t{t}'] = window_row[col]
                    
                    # Add aggregated features
                    if include_aggregated:
                        for col in feature_cols:
                            for method in aggregation_methods:
                                if method == 'mean':
                                    row[f'{col}_{method}'] = window_data[col].mean()
                                elif method == 'std':
                                    row[f'{col}_{method}'] = window_data[col].std()
                                elif method == 'min':
                                    row[f'{col}_{method}'] = window_data[col].min()
                                elif method == 'max':
                                    row[f'{col}_{method}'] = window_data[col].max()
                                elif method == 'median':
                                    row[f'{col}_{method}'] = window_data[col].median()
                    
                    # Add trend-based features (differences, slopes)
                    if include_raw_windows and window_size > 1:
                        for col in feature_cols:
                            values = window_data[col].values
                            
                            # Check for valid data
                            if len(values) < 2:
                                row[f'{col}_first_diff'] = 0
                                row[f'{col}_slope'] = 0
                                row[f'{col}_max_consecutive_change'] = 0
                                continue
                            
                            # First difference (t_n - t_0)
                            try:
                                row[f'{col}_first_diff'] = values[-1] - values[0]
                            except:
                                row[f'{col}_first_diff'] = 0
                            
                            # Linear trend slope - with robust error handling
                            try:
                                # Check for NaN or infinite values
                                if np.any(np.isnan(values)) or np.any(np.isinf(values)):
                                    row[f'{col}_slope'] = 0
                                # Check if all values are the same (no variance)
                                elif np.all(values == values[0]):
                                    row[f'{col}_slope'] = 0
                                else:
                                    x = np.arange(len(values))
                                    slope = np.polyfit(x, values, 1)[0]
                                    # Check if slope result is valid
                                    if np.isnan(slope) or np.isinf(slope):
                                        row[f'{col}_slope'] = 0
                                    else:
                                        row[f'{col}_slope'] = slope
                            except (np.linalg.LinAlgError, ValueError, RuntimeWarning):
                                row[f'{col}_slope'] = 0
                            
                            # Max change between consecutive points
                            try:
                                consecutive_diffs = np.diff(values)
                                if len(consecutive_diffs) > 0:
                                    max_change = np.max(np.abs(consecutive_diffs))
                                    if np.isnan(max_change) or np.isinf(max_change):
                                        row[f'{col}_max_consecutive_change'] = 0
                                    else:
                                        row[f'{col}_max_consecutive_change'] = max_change
                                else:
                                    row[f'{col}_max_consecutive_change'] = 0
                            except:
                                row[f'{col}_max_consecutive_change'] = 0
                    
                    processed_rows.append(row)
        
        self.processed_data = pd.DataFrame(processed_rows)
        return self.processed_data
                    
        # processed_rows.append(row)
        
        # self.processed_data = pd.DataFrame(processed_rows)
        # return self.processed_data
    
    def get_label_options(self) -> Dict[str, Union[pd.Series, np.ndarray]]:
        """
        Get different label encoding options for various ML approaches.
        
        Returns:
            Dictionary with different label encodings
        """
        if self.processed_data is None:
            raise ValueError("Must call create_moving_windows() first")
        
        # Binary transition labels
        binary_labels = self.processed_data['is_transition']
        
        # Multi-class labels (4-class problem)
        multiclass_labels = self.processed_data['transition_category'].astype('category')
        
        # Transition-only labels (excluding no-transition cases)
        transition_mask = self.processed_data['is_transition'] == 1
        transition_only_labels = self.processed_data[transition_mask]['transition_category'].astype('category')
        
        # Create encoded versions
        from sklearn.preprocessing import LabelEncoder
        le_multiclass = LabelEncoder()
        multiclass_encoded = le_multiclass.fit_transform(multiclass_labels)
        
        le_transition = LabelEncoder()
        transition_encoded = le_transition.fit_transform(transition_only_labels) if len(transition_only_labels) > 0 else np.array([])
        
        return {
            'binary': binary_labels,  # 0=no transition, 1=transition
            'multiclass_raw': multiclass_labels,  # wake_to_nrem, nrem_to_wake, stay_wake, stay_nrem
            'multiclass_encoded': multiclass_encoded,  # 0, 1, 2, 3
            'multiclass_classes': le_multiclass.classes_,
            'transition_only_raw': transition_only_labels,  # wake_to_nrem, nrem_to_wake (only transition cases)
            'transition_only_encoded': transition_encoded,  # 0, 1 (only transition cases)
            'transition_classes': le_transition.classes_ if len(transition_only_labels) > 0 else [],
            'transition_mask': transition_mask  # Boolean mask for transition cases
        }
    
    def create_balanced_splits(self, split_type: str = 'transductive', 
                             label_type: str = 'multiclass',
                             test_size: float = 0.2, 
                             animal: Optional[str] = None,
                             random_state: int = 42,
                             stratify: bool = True) -> Dict:
        """
        Create stratified splits that balance different transition types.
        
        Args:
            split_type: 'transductive' or 'inductive'
            label_type: 'binary', 'multiclass', or 'transition_only'
            test_size: Proportion for test set
            animal: Specific animal (for transductive splits)
            random_state: Random seed
            stratify: Whether to stratify by label distribution
            
        Returns:
            Dictionary with train/test splits
        """
        if self.processed_data is None:
            raise ValueError("Must call create_moving_windows() first")
        
        # Get appropriate labels
        label_options = self.get_label_options()
        
        if label_type == 'binary':
            y = label_options['binary']
            data_mask = np.ones(len(self.processed_data), dtype=bool)
        elif label_type == 'multiclass':
            y = label_options['multiclass_encoded']
            data_mask = np.ones(len(self.processed_data), dtype=bool)
        elif label_type == 'transition_only':
            y = label_options['transition_only_encoded']
            data_mask = label_options['transition_mask']
        else:
            raise ValueError("label_type must be 'binary', 'multiclass', or 'transition_only'")
        
        # Filter data if needed
        filtered_data = self.processed_data[data_mask] if not data_mask.all() else self.processed_data
        
        # Perform split based on type
        if split_type == 'transductive':
            return self._transductive_split_balanced(filtered_data, y, test_size, animal, random_state, stratify)
        else:
            return self._inductive_split_balanced(filtered_data, y, test_size, random_state, stratify)
    
    def _transductive_split_balanced(self, data, y, test_size, animal, random_state, stratify):
        """Helper method for balanced transductive splits"""
        from sklearn.model_selection import train_test_split
        
        if animal is not None:
            # Split for specific animal
            animal_data = data[data['animal'] == animal]
            blocks = animal_data['block'].unique()
            
            if stratify and len(np.unique(y)) > 1:
                # Try to stratify by block-level label distribution
                block_labels = []
                for block in blocks:
                    block_mask = animal_data['block'] == block
                    block_y = y[data['animal'] == animal][block_mask]
                    # Use most common label in block
                    block_labels.append(np.bincount(block_y).argmax())
                
                try:
                    train_blocks, test_blocks = train_test_split(
                        blocks, test_size=test_size, random_state=random_state,
                        stratify=block_labels
                    )
                except:
                    # Fall back to non-stratified if stratification fails
                    train_blocks, test_blocks = train_test_split(
                        blocks, test_size=test_size, random_state=random_state
                    )
            else:
                train_blocks, test_blocks = train_test_split(
                    blocks, test_size=test_size, random_state=random_state
                )
            
            train_mask = data['animal'] == animal
            train_mask = train_mask & data['block'].isin(train_blocks)
            test_mask = data['animal'] == animal
            test_mask = test_mask & data['block'].isin(test_blocks)
        
        else:
            # Pool all animals - implement similar logic for all animals
            train_indices = []
            test_indices = []
            
            for animal_id in data['animal'].unique():
                animal_mask = data['animal'] == animal_id
                animal_data = data[animal_mask]
                blocks = animal_data['block'].unique()
                
                if len(blocks) == 1:
                    train_indices.extend(data[animal_mask].index.tolist())
                else:
                    train_blocks, test_blocks = train_test_split(
                        blocks, test_size=test_size, random_state=random_state
                    )
                    
                    train_indices.extend(data[animal_mask & data['block'].isin(train_blocks)].index.tolist())
                    test_indices.extend(data[animal_mask & data['block'].isin(test_blocks)].index.tolist())
            
            train_mask = data.index.isin(train_indices)
            test_mask = data.index.isin(test_indices)
        
        return self._create_split_dict(data, y, train_mask, test_mask)
    
    def _inductive_split_balanced(self, data, y, test_size, random_state, stratify):
        """Helper method for balanced inductive splits"""
        from sklearn.model_selection import train_test_split
        
        animals = data['animal'].unique()
        
        if stratify and len(animals) > 1:
            # Stratify by animal-level label distribution
            animal_labels = []
            for animal in animals:
                animal_mask = data['animal'] == animal
                animal_y = y[animal_mask]
                # Use most common label for this animal
                animal_labels.append(np.bincount(animal_y).argmax())
            
            try:
                train_animals, test_animals = train_test_split(
                    animals, test_size=test_size, random_state=random_state,
                    stratify=animal_labels
                )
            except:
                # Fall back to non-stratified
                train_animals, test_animals = train_test_split(
                    animals, test_size=test_size, random_state=random_state
                )
        else:
            train_animals, test_animals = train_test_split(
                animals, test_size=test_size, random_state=random_state
            )
        
        train_mask = data['animal'].isin(train_animals)
        test_mask = data['animal'].isin(test_animals)
        
        result = self._create_split_dict(data, y, train_mask, test_mask)
        result['train_animals'] = train_animals
        result['test_animals'] = test_animals
        
        return result
    
    def _create_split_dict(self, data, y, train_mask, test_mask):
        """Helper to create consistent split dictionaries"""
        # Get features
        feature_cols = [col for col in data.columns 
                       if col not in ['animal', 'block', 'window_end_index', 'current_state', 
                                    'is_transition', 'transition_type', 'transition_category', 'previous_state']]
        
        X = data[feature_cols]
        
        return {
            'X_train': X[train_mask],
            'X_test': X[test_mask],
            'y_train': y[train_mask],
            'y_test': y[test_mask],
            'train_metadata': data[train_mask][['animal', 'block', 'window_end_index', 'transition_type']],
            'test_metadata': data[test_mask][['animal', 'block', 'window_end_index', 'transition_type']]
        }
    def get_feature_label_split(self, label_type: str = 'binary') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get features (X) and labels (y) from processed data.
        
        Args:
            label_type: 'binary', 'multiclass', or 'transition_only'
        
        Returns:
            Tuple of (features_df, labels_series)
        """
        if self.processed_data is None:
            raise ValueError("Must call create_moving_windows() first")
        
        # Features are all columns except metadata and label columns
        feature_cols = [col for col in self.processed_data.columns 
                       if col not in ['animal', 'block', 'window_end_index', 'current_state', 
                                    'is_transition', 'transition_type', 'transition_category', 'previous_state']]
        
        X = self.processed_data[feature_cols]
        
        label_options = self.get_label_options()
        
        if label_type == 'binary':
            y = label_options['binary']
        elif label_type == 'multiclass':
            y = pd.Series(label_options['multiclass_encoded'], index=X.index)
        elif label_type == 'transition_only':
            # Filter to only transition cases
            mask = label_options['transition_mask']
            X = X[mask]
            y = pd.Series(label_options['transition_only_encoded'], index=X.index)
        else:
            raise ValueError("label_type must be 'binary', 'multiclass', or 'transition_only'")
        
        return X, y
    
    def transductive_split(self, test_size: float = 0.2, 
                          animal: Optional[str] = None,
                          random_state: int = 42) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """
        Create transductive split (block-based split within animals).
        
        Args:
            test_size: Proportion of blocks to use for testing
            animal: Specific animal to split (if None, pools all animals)
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with 'X_train', 'X_test', 'y_train', 'y_test'
        """
        if self.processed_data is None:
            raise ValueError("Must call create_moving_windows() first")
        
        if animal is not None:
            # Split for specific animal
            animal_data = self.processed_data[self.processed_data['animal'] == animal]
            blocks = animal_data['block'].unique()
            
            train_blocks, test_blocks = train_test_split(
                blocks, test_size=test_size, random_state=random_state
            )
            
            train_data = animal_data[animal_data['block'].isin(train_blocks)]
            test_data = animal_data[animal_data['block'].isin(test_blocks)]
            
        else:
            # Pool all animals and split blocks proportionally
            train_data_list = []
            test_data_list = []
            
            for animal_id in self.processed_data['animal'].unique():
                animal_data = self.processed_data[self.processed_data['animal'] == animal_id]
                blocks = animal_data['block'].unique()
                
                if len(blocks) == 1:
                    # If only one block, put it in train
                    train_data_list.append(animal_data)
                else:
                    train_blocks, test_blocks = train_test_split(
                        blocks, test_size=test_size, random_state=random_state
                    )
                    
                    train_data_list.append(animal_data[animal_data['block'].isin(train_blocks)])
                    test_data_list.append(animal_data[animal_data['block'].isin(test_blocks)])
            
            train_data = pd.concat(train_data_list, ignore_index=True) if train_data_list else pd.DataFrame()
            test_data = pd.concat(test_data_list, ignore_index=True) if test_data_list else pd.DataFrame()
        
        # Get features and labels
        X, y = self.get_feature_label_split()
        
        train_indices = train_data.index if not train_data.empty else []
        test_indices = test_data.index if not test_data.empty else []
        
        return {
            'X_train': X.loc[train_indices],
            'X_test': X.loc[test_indices],
            'y_train': y.loc[train_indices],
            'y_test': y.loc[test_indices],
            'train_metadata': train_data[['animal', 'block', 'window_end_index']],
            'test_metadata': test_data[['animal', 'block', 'window_end_index']]
        }
    
    def inductive_split(self, test_size: float = 0.2, 
                       random_state: int = 42) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
        """
        Create inductive split (animal-based split).
        
        Args:
            test_size: Proportion of animals to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with 'X_train', 'X_test', 'y_train', 'y_test'
        """
        if self.processed_data is None:
            raise ValueError("Must call create_moving_windows() first")
        
        animals = self.processed_data['animal'].unique()
        train_animals, test_animals = train_test_split(
            animals, test_size=test_size, random_state=random_state
        )
        
        train_data = self.processed_data[self.processed_data['animal'].isin(train_animals)]
        test_data = self.processed_data[self.processed_data['animal'].isin(test_animals)]
        
        # Get features and labels
        X, y = self.get_feature_label_split()
        
        return {
            'X_train': X.loc[train_data.index],
            'X_test': X.loc[test_data.index],
            'y_train': y.loc[train_data.index],
            'y_test': y.loc[test_data.index],
            'train_metadata': train_data[['animal', 'block', 'window_end_index']],
            'test_metadata': test_data[['animal', 'block', 'window_end_index']],
            'train_animals': train_animals,
            'test_animals': test_animals
        }
    
    def create_sequence_data(self, window_size: int = 5, 
                           feature_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Create sequence data suitable for RNNs/LSTMs where each sample is a sequence.
        
        Args:
            window_size: Size of the moving window
            feature_cols: List of feature columns to use
            
        Returns:
            Tuple of (X_sequences, y_labels, metadata_df)
            - X_sequences: 3D array of shape (n_samples, window_size, n_features)
            - y_labels: 1D array of transition labels
            - metadata_df: DataFrame with animal, block, window_end_index info
        """
        if feature_cols is None:
            feature_cols = self.feature_columns
            
        sequences = []
        labels = []
        metadata = []
        
        for animal in self.df['animal'].unique():
            animal_data = self.df[self.df['animal'] == animal]
            
            for block in animal_data['block'].unique():
                block_data = animal_data[animal_data['block'] == block].reset_index(drop=True)
                
                # Create sequences for this block
                for i in range(window_size - 1, len(block_data)):
                    window_data = block_data.iloc[i - window_size + 1:i + 1]
                    
                    # Extract sequence (window_size x n_features)
                    sequence = window_data[feature_cols].values
                    sequences.append(sequence)
                    
                    # Determine transition label and type
                    if i > 0:
                        prev_state = block_data.iloc[i-1]['state']
                        current_state = block_data.iloc[i]['state']
                        
                        if prev_state != current_state:
                            if prev_state.lower() in ['wake', 'w'] and current_state.lower() in ['nrem', 'n']:
                                transition_type = 'wake_to_nrem'
                            elif prev_state.lower() in ['nrem', 'n'] and current_state.lower() in ['wake', 'w']:
                                transition_type = 'nrem_to_wake'
                            else:
                                transition_type = f'{prev_state}_to_{current_state}'
                            labels.append(1)
                        else:
                            if current_state.lower() in ['wake', 'w']:
                                transition_type = 'stay_wake'
                            elif current_state.lower() in ['nrem', 'n']:
                                transition_type = 'stay_nrem'
                            else:
                                transition_type = f'stay_{current_state}'
                            labels.append(0)
                    else:
                        current_state = block_data.iloc[i]['state']
                        transition_type = f'initial_{current_state}'
                        labels.append(0)
                    
                    # Store metadata
                    metadata.append({
                        'animal': animal,
                        'block': block,
                        'window_end_index': i,
                        'current_state': block_data.iloc[i]['state'],
                        'previous_state': block_data.iloc[i-1]['state'] if i > 0 else None,
                        'transition_type': transition_type
                    })
        
        return np.array(sequences), np.array(labels), pd.DataFrame(metadata)
        """
        Get summary statistics of the processed data.
        
        Returns:
            Dictionary with summary information
        """
        if self.processed_data is None:
            return {"error": "Must call create_moving_windows() first"}
        
        summary = {
            'total_windows': len(self.processed_data),
            'num_animals': self.processed_data['animal'].nunique(),
            'num_blocks': self.processed_data.groupby('animal')['block'].nunique().to_dict(),
            'transition_rate': self.processed_data['is_transition'].mean(),
            'transitions_per_animal': self.processed_data.groupby('animal')['is_transition'].sum().to_dict(),
            'transition_types_per_animal': self.processed_data.groupby(['animal', 'transition_category']).size().unstack(fill_value=0).to_dict(),
            'overall_transition_distribution': self.processed_data['transition_category'].value_counts().to_dict(),
            'windows_per_animal': self.processed_data.groupby('animal').size().to_dict(),
            'feature_columns': [col for col in self.processed_data.columns 
                              if col not in ['animal', 'block', 'window_end_index', 'current_state', 'is_transition']]
        }
        
    def sequence_based_splits(self, X_sequences: np.ndarray, y_labels: np.ndarray, 
                            metadata_df: pd.DataFrame, split_type: str = 'transductive',
                            test_size: float = 0.2, animal: Optional[str] = None,
                            random_state: int = 42) -> Dict:
        """
        Create train/test splits for sequence data.
        
        Args:
            X_sequences: 3D sequence array from create_sequence_data()
            y_labels: Labels array from create_sequence_data()
            metadata_df: Metadata DataFrame from create_sequence_data()
            split_type: 'transductive' or 'inductive'
            test_size: Proportion for test set
            animal: Specific animal (for transductive splits)
            random_state: Random seed
            
        Returns:
            Dictionary with train/test splits for sequences
        """
        if split_type == 'transductive':
            if animal is not None:
                # Split for specific animal
                animal_mask = metadata_df['animal'] == animal
                animal_data = metadata_df[animal_mask]
                blocks = animal_data['block'].unique()
                
                train_blocks, test_blocks = train_test_split(
                    blocks, test_size=test_size, random_state=random_state
                )
                
                train_mask = animal_mask & metadata_df['block'].isin(train_blocks)
                test_mask = animal_mask & metadata_df['block'].isin(test_blocks)
                
            else:
                # Pool all animals
                train_masks = []
                test_masks = []
                
                for animal_id in metadata_df['animal'].unique():
                    animal_mask = metadata_df['animal'] == animal_id
                    animal_data = metadata_df[animal_mask]
                    blocks = animal_data['block'].unique()
                    
                    if len(blocks) == 1:
                        train_masks.append(animal_mask)
                    else:
                        train_blocks, test_blocks = train_test_split(
                            blocks, test_size=test_size, random_state=random_state
                        )
                        
                        train_masks.append(animal_mask & metadata_df['block'].isin(train_blocks))
                        test_masks.append(animal_mask & metadata_df['block'].isin(test_blocks))
                
                train_mask = np.logical_or.reduce(train_masks) if train_masks else np.zeros(len(metadata_df), dtype=bool)
                test_mask = np.logical_or.reduce(test_masks) if test_masks else np.zeros(len(metadata_df), dtype=bool)
                
        else:  # inductive
            animals = metadata_df['animal'].unique()
            train_animals, test_animals = train_test_split(
                animals, test_size=test_size, random_state=random_state
            )
            
            train_mask = metadata_df['animal'].isin(train_animals)
            test_mask = metadata_df['animal'].isin(test_animals)
        
        return {
            'X_train': X_sequences[train_mask],
            'X_test': X_sequences[test_mask],
            'y_train': y_labels[train_mask],
            'y_test': y_labels[test_mask],
            'train_metadata': metadata_df[train_mask],
            'test_metadata': metadata_df[test_mask]
        }