"""
CyberCore-QC: Fuzzy Inference System (FIS)
===========================================
Scikit-Fuzzy based decision system for quality control severity assessment.
Takes CNN defect probability + material properties as input.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from typing import Dict, Tuple, List
import pickle


class FuzzyQualityController:
    """
    Fuzzy Inference System for Quality Control Decision Making.
    
    Inputs:
    - defect_probability: Probability of defect from CNN (0.0 - 1.0)
    - material_fragility: Material fragility index (0.0 - 1.0)
    
    Output:
    - severity_score: Quality severity score (0.0 - 10.0)
      - 0-3: Accept (Green zone)
      - 3-7: Rework (Yellow zone)
      - 7-10: Reject (Red zone)
    """
    
    def __init__(self):
        """Initialize the Fuzzy Inference System."""
        
        # Create fuzzy variables
        self._create_fuzzy_variables()
        
        # Create membership functions
        self._create_membership_functions()
        
        # Create fuzzy rules
        self._create_fuzzy_rules()
        
        # Create control system
        self._create_control_system()
        
    def _create_fuzzy_variables(self):
        """Create input and output fuzzy variables."""
        
        # Input 1: Defect Probability from CNN (0.0 to 1.0)
        self.defect_prob = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'defect_probability')
        
        # Input 2: Material Fragility (0.0 to 1.0)
        self.material_fragility = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'material_fragility')
        
        # Output: Severity Score (0.0 to 10.0)
        self.severity = ctrl.Consequent(np.arange(0, 10.01, 0.1), 'severity_score')
        
    def _create_membership_functions(self):
        """
        Create membership functions for fuzzy variables.
        These will be optimized by the Genetic Algorithm.
        """
        
        # Defect Probability Membership Functions
        # Low: 0.0 - 0.4 (triangular)
        self.defect_prob['low'] = fuzz.trimf(self.defect_prob.universe, [0.0, 0.0, 0.4])
        
        # Medium: 0.2 - 0.7 (triangular)
        self.defect_prob['medium'] = fuzz.trimf(self.defect_prob.universe, [0.2, 0.5, 0.7])
        
        # High: 0.6 - 1.0 (triangular)
        self.defect_prob['high'] = fuzz.trimf(self.defect_prob.universe, [0.6, 1.0, 1.0])
        
        # Material Fragility Membership Functions
        # Low: 0.0 - 0.4
        self.material_fragility['low'] = fuzz.trimf(self.material_fragility.universe, [0.0, 0.0, 0.4])
        
        # Medium: 0.3 - 0.7
        self.material_fragility['medium'] = fuzz.trimf(self.material_fragility.universe, [0.3, 0.5, 0.7])
        
        # High: 0.6 - 1.0
        self.material_fragility['high'] = fuzz.trimf(self.material_fragility.universe, [0.6, 1.0, 1.0])
        
        # Severity Score Membership Functions (Output)
        # Accept: 0 - 3 (Green zone)
        self.severity['accept'] = fuzz.trimf(self.severity.universe, [0, 0, 3])
        
        # Rework: 2 - 8 (Yellow zone)
        self.severity['rework'] = fuzz.trimf(self.severity.universe, [2, 5, 8])
        
        # Reject: 7 - 10 (Red zone)
        self.severity['reject'] = fuzz.trimf(self.severity.universe, [7, 10, 10])
        
    def _create_fuzzy_rules(self):
        """
        Create fuzzy rules for decision making.
        
        Rule Logic:
        - Low defect prob + Low fragility -> Accept
        - High defect prob + High fragility -> Reject
        - Mixed conditions -> Rework or intermediate decisions
        """
        
        self.rules = [
            # Low defect probability rules
            ctrl.Rule(self.defect_prob['low'] & self.material_fragility['low'], self.severity['accept']),
            ctrl.Rule(self.defect_prob['low'] & self.material_fragility['medium'], self.severity['accept']),
            ctrl.Rule(self.defect_prob['low'] & self.material_fragility['high'], self.severity['rework']),
            
            # Medium defect probability rules
            ctrl.Rule(self.defect_prob['medium'] & self.material_fragility['low'], self.severity['rework']),
            ctrl.Rule(self.defect_prob['medium'] & self.material_fragility['medium'], self.severity['rework']),
            ctrl.Rule(self.defect_prob['medium'] & self.material_fragility['high'], self.severity['reject']),
            
            # High defect probability rules
            ctrl.Rule(self.defect_prob['high'] & self.material_fragility['low'], self.severity['rework']),
            ctrl.Rule(self.defect_prob['high'] & self.material_fragility['medium'], self.severity['reject']),
            ctrl.Rule(self.defect_prob['high'] & self.material_fragility['high'], self.severity['reject']),
        ]
        
    def _create_control_system(self):
        """Create the fuzzy control system."""
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)
        
    def predict(self, defect_probability: float, material_fragility: float) -> Dict[str, float]:
        """
        Make a fuzzy decision.
        
        Args:
            defect_probability: Defect probability from CNN (0.0 - 1.0)
            material_fragility: Material fragility index (0.0 - 1.0)
            
        Returns:
            Dictionary with severity score and decision
        """
        # Ensure inputs are within valid range
        defect_probability = np.clip(defect_probability, 0.0, 1.0)
        material_fragility = np.clip(material_fragility, 0.0, 1.0)
        
        # Set inputs
        self.simulation.input['defect_probability'] = defect_probability
        self.simulation.input['material_fragility'] = material_fragility
        
        # Compute output
        self.simulation.compute()
        
        severity_score = self.simulation.output['severity_score']
        
        # Determine decision
        if severity_score < 3:
            decision = 'Accept'
            color = 'green'
        elif severity_score < 7:
            decision = 'Rework'
            color = 'yellow'
        else:
            decision = 'Reject'
            color = 'red'
        
        return {
            'severity_score': severity_score,
            'decision': decision,
            'color': color,
            'defect_probability': defect_probability,
            'material_fragility': material_fragility
        }
    
    def batch_predict(self, defect_probs: List[float], 
                     material_fragilities: List[float]) -> List[Dict]:
        """
        Make predictions for a batch of samples.
        
        Args:
            defect_probs: List of defect probabilities
            material_fragilities: List of material fragility values
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for defect_prob, mat_frag in zip(defect_probs, material_fragilities):
            result = self.predict(defect_prob, mat_frag)
            results.append(result)
        
        return results
    
    def get_membership_params(self) -> Dict:
        """
        Extract current membership function parameters.
        
        Returns:
            Dictionary of membership function parameters (for GA optimization)
        """
        params = {
            'defect_prob_low': self.defect_prob['low'].mf.tolist(),
            'defect_prob_medium': self.defect_prob['medium'].mf.tolist(),
            'defect_prob_high': self.defect_prob['high'].mf.tolist(),
            'material_fragility_low': self.material_fragility['low'].mf.tolist(),
            'material_fragility_medium': self.material_fragility['medium'].mf.tolist(),
            'material_fragility_high': self.material_fragility['high'].mf.tolist(),
            'severity_accept': self.severity['accept'].mf.tolist(),
            'severity_rework': self.severity['rework'].mf.tolist(),
            'severity_reject': self.severity['reject'].mf.tolist(),
        }
        
        return params
    
    def set_membership_params(self, params: Dict):
        """
        Set membership function parameters.
        Used by GA to update fuzzy sets during optimization.
        
        Args:
            params: Dictionary of membership function parameters
        """
        # Update defect probability membership functions
        if 'defect_prob_low' in params:
            self.defect_prob['low'].mf = np.array(params['defect_prob_low'])
        if 'defect_prob_medium' in params:
            self.defect_prob['medium'].mf = np.array(params['defect_prob_medium'])
        if 'defect_prob_high' in params:
            self.defect_prob['high'].mf = np.array(params['defect_prob_high'])
        
        # Update material fragility membership functions
        if 'material_fragility_low' in params:
            self.material_fragility['low'].mf = np.array(params['material_fragility_low'])
        if 'material_fragility_medium' in params:
            self.material_fragility['medium'].mf = np.array(params['material_fragility_medium'])
        if 'material_fragility_high' in params:
            self.material_fragility['high'].mf = np.array(params['material_fragility_high'])
        
        # Update severity membership functions
        if 'severity_accept' in params:
            self.severity['accept'].mf = np.array(params['severity_accept'])
        if 'severity_rework' in params:
            self.severity['rework'].mf = np.array(params['severity_rework'])
        if 'severity_reject' in params:
            self.severity['reject'].mf = np.array(params['severity_reject'])
    
    def get_triangular_params(self) -> Dict[str, List[float]]:
        """
        Get triangular membership function parameters (a, b, c).
        These are the parameters that will be optimized by GA.
        
        Returns:
            Dictionary mapping variable names to [a, b, c] parameters
        """
        params = {}
        
        # For triangular membership functions, we need to extract the peak points
        # This is what GA will optimize
        
        # Defect Probability
        params['defect_prob_low'] = [0.0, 0.0, 0.4]
        params['defect_prob_medium'] = [0.2, 0.5, 0.7]
        params['defect_prob_high'] = [0.6, 1.0, 1.0]
        
        # Material Fragility
        params['material_fragility_low'] = [0.0, 0.0, 0.4]
        params['material_fragility_medium'] = [0.3, 0.5, 0.7]
        params['material_fragility_high'] = [0.6, 1.0, 1.0]
        
        # Severity (not optimized, kept fixed)
        params['severity_accept'] = [0, 0, 3]
        params['severity_rework'] = [2, 5, 8]
        params['severity_reject'] = [7, 10, 10]
        
        return params
    
    def update_from_triangular_params(self, params: Dict[str, List[float]]):
        """
        Update membership functions from triangular parameters.
        
        Args:
            params: Dictionary of [a, b, c] parameters for each membership function
        """
        # Update defect probability
        if 'defect_prob_low' in params:
            a, b, c = params['defect_prob_low']
            self.defect_prob['low'].mf = fuzz.trimf(self.defect_prob.universe, [a, b, c])
        
        if 'defect_prob_medium' in params:
            a, b, c = params['defect_prob_medium']
            self.defect_prob['medium'].mf = fuzz.trimf(self.defect_prob.universe, [a, b, c])
        
        if 'defect_prob_high' in params:
            a, b, c = params['defect_prob_high']
            self.defect_prob['high'].mf = fuzz.trimf(self.defect_prob.universe, [a, b, c])
        
        # Update material fragility
        if 'material_fragility_low' in params:
            a, b, c = params['material_fragility_low']
            self.material_fragility['low'].mf = fuzz.trimf(self.material_fragility.universe, [a, b, c])
        
        if 'material_fragility_medium' in params:
            a, b, c = params['material_fragility_medium']
            self.material_fragility['medium'].mf = fuzz.trimf(self.material_fragility.universe, [a, b, c])
        
        if 'material_fragility_high' in params:
            a, b, c = params['material_fragility_high']
            self.material_fragility['high'].mf = fuzz.trimf(self.material_fragility.universe, [a, b, c])
        
        # Recreate control system with updated membership functions
        self._create_control_system()
    
    def save(self, filepath: str):
        """Save the FIS to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.get_triangular_params(), f)
    
    def load(self, filepath: str):
        """Load the FIS from a file."""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        self.update_from_triangular_params(params)


if __name__ == "__main__":
    # Test the Fuzzy Inference System
    fis = FuzzyQualityController()
    
    # Test predictions
    test_cases = [
        (0.1, 0.2, "Low defect, low fragility"),
        (0.5, 0.5, "Medium defect, medium fragility"),
        (0.9, 0.8, "High defect, high fragility"),
        (0.3, 0.7, "Low defect, high fragility"),
        (0.8, 0.2, "High defect, low fragility"),
    ]
    
    print("Fuzzy Inference System Test Cases:")
    print("=" * 70)
    
    for defect_prob, mat_frag, description in test_cases:
        result = fis.predict(defect_prob, mat_frag)
        print(f"\n{description}")
        print(f"  Defect Prob: {defect_prob:.2f}, Material Fragility: {mat_frag:.2f}")
        print(f"  → Severity: {result['severity_score']:.2f}")
        print(f"  → Decision: {result['decision']} ({result['color']})")
    
    print("\n" + "=" * 70)
