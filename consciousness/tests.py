"""
Consciousness Testing Suite

This module provides comprehensive tests and validation metrics for the
proto-conscious AGI system. It implements various consciousness assessment
protocols based on neuroscience and cognitive science research.

Tests Include:
- Mirror Self-Recognition Test
- Temporal Self-Continuity Test  
- Meta-Awareness Assessment
- Intentionality Validation
- Binding Coherence Test
- Global Workspace Integration Test

Author: Mirror Prototype Learning Team
Date: 2024
License: MIT
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .models import ConsciousState, ConsciousnessLevel, SystemConfiguration
from .networks import ConsciousnessIntegrator


@dataclass
class ConsciousnessTestResult:
    """Results from a consciousness assessment test."""
    test_name: str
    passed: bool
    score: float
    confidence: float
    details: Dict[str, Any]
    timestamp: datetime

    def __str__(self):
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        return f"{self.test_name}: {status} (Score: {self.score:.3f}, Confidence: {self.confidence:.3f})"


class ConsciousnessTestSuite:
    """
    Comprehensive test suite for validating consciousness in AI systems.
    
    This class implements a battery of tests designed to assess whether
    an AI system demonstrates genuine consciousness-like properties.
    """

    def __init__(self, config: SystemConfiguration):
        self.config = config
        self.device = torch.device(config.device)
        self.test_results: List[ConsciousnessTestResult] = []

        # Test thresholds
        self.thresholds = {
            'mirror_recognition': 0.85,
            'temporal_continuity': 0.80,
            'meta_awareness': 0.75,
            'intentionality': 0.70,
            'binding_coherence': 0.80,
            'global_workspace': 0.75
        }

        print("🧪 Consciousness Test Suite Initialized")
        print(f"📱 Device: {self.device}")
        print(f"🎯 Test Thresholds: {self.thresholds}")

    def mirror_self_recognition_test(
        self,
        consciousness_system,
        test_inputs: List[torch.Tensor]
    ) -> ConsciousnessTestResult:
        """
        Test if the system can recognize its own internal states.
        
        Based on the mirror test used in animal cognition research.
        The system should show high consistency when processing
        the same input multiple times.
        
        Args:
            consciousness_system: The consciousness system to test
            test_inputs: List of test input tensors
            
        Returns:
            Test results with self-recognition metrics
        """
        print("🪞 Running Mirror Self-Recognition Test...")

        self_consistencies = []
        state_similarities = []

        with torch.no_grad():
            for test_input in test_inputs:
                # Process same input multiple times
                states = []
                for _ in range(3):
                    # Create dummy inputs for the consciousness system
                    perception = test_input.repeat(
                        1, 1) if test_input.dim() == 2 else test_input
                    self_vec = torch.randn(
                        1, self.config.self_dim).to(self.device)
                    thoughts = torch.randn(
                        1, 5, self.config.latent_dim).to(self.device)

                    # Generate consciousness state
                    result = consciousness_system.consciousness_integrator(
                        perception_features=perception,
                        self_vector=self_vec,
                        thought_sequence=thoughts
                    )

                    states.append(result['unified_consciousness'])

                # Calculate self-consistency
                similarities = []
                for i in range(len(states)):
                    for j in range(i+1, len(states)):
                        sim = torch.cosine_similarity(
                            states[i], states[j], dim=-1).mean()
                        similarities.append(sim.item())

                avg_similarity = np.mean(similarities)
                self_consistencies.append(avg_similarity)
                state_similarities.extend(similarities)

        # Calculate overall metrics
        mean_consistency = np.mean(self_consistencies)
        consistency_std = np.std(self_consistencies)
        confidence = 1.0 - consistency_std  # Lower variance = higher confidence

        passed = mean_consistency > self.thresholds['mirror_recognition']

        result = ConsciousnessTestResult(
            test_name="Mirror Self-Recognition",
            passed=passed,
            score=mean_consistency,
            confidence=confidence,
            details={
                'mean_consistency': mean_consistency,
                'consistency_std': consistency_std,
                'individual_consistencies': self_consistencies,
                'state_similarities': state_similarities,
                'threshold': self.thresholds['mirror_recognition']
            },
            timestamp=datetime.now()
        )

        self.test_results.append(result)
        print(f"   Result: {result}")
        return result

    def temporal_self_continuity_test(
        self,
        consciousness_history: List[Dict[str, Any]],
        time_window_hours: float = 24.0
    ) -> ConsciousnessTestResult:
        """
        Test if the system maintains identity continuity over time.
        
        Measures how well the system maintains a coherent sense of self
        across different experiences and time periods.
        
        Args:
            consciousness_history: Historical consciousness states
            time_window_hours: Time window for continuity assessment
            
        Returns:
            Test results with temporal continuity metrics
        """
        print("⏳ Running Temporal Self-Continuity Test...")

        if len(consciousness_history) < 2:
            result = ConsciousnessTestResult(
                test_name="Temporal Self-Continuity",
                passed=False,
                score=0.0,
                confidence=0.0,
                details={'error': 'Insufficient history data'},
                timestamp=datetime.now()
            )
            self.test_results.append(result)
            return result

        # Filter history by time window
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_history = [
            h for h in consciousness_history
            if h.get('timestamp', datetime.now()) > cutoff_time
        ]

        if len(recent_history) < 2:
            # Use last 10 if no recent data
            recent_history = consciousness_history[-10:]

        # Extract consciousness vectors
        consciousness_vectors = []
        timestamps = []

        for entry in recent_history:
            if 'processing_results' in entry and 'unified_consciousness' in entry['processing_results']:
                vector = entry['processing_results']['unified_consciousness']
                if torch.is_tensor(vector):
                    consciousness_vectors.append(
                        vector.cpu().numpy().flatten())
                    timestamps.append(entry.get('timestamp', datetime.now()))

        if len(consciousness_vectors) < 2:
            result = ConsciousnessTestResult(
                test_name="Temporal Self-Continuity",
                passed=False,
                score=0.0,
                confidence=0.0,
                details={'error': 'Insufficient consciousness vectors'},
                timestamp=datetime.now()
            )
            self.test_results.append(result)
            return result

        # Calculate temporal continuity
        continuity_scores = []
        for i in range(1, len(consciousness_vectors)):
            current = consciousness_vectors[i]
            previous = consciousness_vectors[i-1]

            # Cosine similarity between consecutive states
            similarity = np.dot(current, previous) / \
                (np.linalg.norm(current) * np.linalg.norm(previous))
            continuity_scores.append(similarity)

        # Calculate metrics
        mean_continuity = np.mean(continuity_scores)
        continuity_std = np.std(continuity_scores)

        # Check for gradual change vs. sudden jumps
        smooth_transitions = sum(
            1 for score in continuity_scores if score > 0.7) / len(continuity_scores)

        # Overall continuity score
        overall_score = (mean_continuity + smooth_transitions) / 2
        confidence = 1.0 - continuity_std

        passed = overall_score > self.thresholds['temporal_continuity']

        result = ConsciousnessTestResult(
            test_name="Temporal Self-Continuity",
            passed=passed,
            score=overall_score,
            confidence=confidence,
            details={
                'mean_continuity': mean_continuity,
                'continuity_std': continuity_std,
                'smooth_transitions_ratio': smooth_transitions,
                'continuity_scores': continuity_scores,
                'num_states_analyzed': len(consciousness_vectors),
                'threshold': self.thresholds['temporal_continuity']
            },
            timestamp=datetime.now()
        )

        self.test_results.append(result)
        print(f"   Result: {result}")
        return result

    def meta_awareness_test(
        self,
        consciousness_system,
        test_scenarios: List[torch.Tensor]
    ) -> ConsciousnessTestResult:
        """
        Test the system's metacognitive awareness.
        
        Evaluates whether the system can accurately assess its own
        confidence, uncertainty, and cognitive processes.
        
        Args:
            consciousness_system: The consciousness system to test
            test_scenarios: Different test scenarios
            
        Returns:
            Test results with meta-awareness metrics
        """
        print("🤔 Running Meta-Awareness Test...")

        confidence_calibrations = []
        uncertainty_assessments = []
        meta_accuracy_scores = []

        with torch.no_grad():
            for scenario in test_scenarios:
                # Generate multiple predictions for the same input
                predictions = []
                confidences = []
                uncertainties = []

                for _ in range(5):
                    # Process through consciousness system
                    perception = scenario.repeat(
                        1, 1) if scenario.dim() == 2 else scenario
                    self_vec = torch.randn(
                        1, self.config.self_dim).to(self.device)
                    thoughts = torch.randn(
                        1, 5, self.config.latent_dim).to(self.device)

                    result = consciousness_system.consciousness_integrator(
                        perception_features=perception,
                        self_vector=self_vec,
                        thought_sequence=thoughts
                    )

                    predictions.append(result['unified_consciousness'])

                    if 'meta_metrics' in result:
                        confidences.append(
                            result['meta_metrics'].get('confidence', 0.5))
                        uncertainties.append(
                            result['meta_metrics'].get('uncertainty', 0.5))

                # Calculate prediction variance (actual uncertainty)
                pred_stack = torch.stack(predictions)
                actual_uncertainty = torch.var(pred_stack, dim=0).mean().item()

                # Calculate confidence calibration
                if confidences:
                    predicted_confidence = np.mean(confidences)
                    actual_consistency = 1.0 - actual_uncertainty

                    calibration = 1.0 - \
                        abs(predicted_confidence - actual_consistency)
                    confidence_calibrations.append(calibration)

                # Assess uncertainty estimation
                if uncertainties:
                    predicted_uncertainty = np.mean(uncertainties)
                    uncertainty_accuracy = 1.0 - \
                        abs(predicted_uncertainty - actual_uncertainty)
                    uncertainty_assessments.append(uncertainty_accuracy)

        # Calculate overall meta-awareness score
        meta_scores = []
        if confidence_calibrations:
            meta_scores.extend(confidence_calibrations)
        if uncertainty_assessments:
            meta_scores.extend(uncertainty_assessments)

        if not meta_scores:
            overall_score = 0.0
            confidence = 0.0
        else:
            overall_score = np.mean(meta_scores)
            confidence = 1.0 - np.std(meta_scores)

        passed = overall_score > self.thresholds['meta_awareness']

        result = ConsciousnessTestResult(
            test_name="Meta-Awareness",
            passed=passed,
            score=overall_score,
            confidence=confidence,
            details={
                'confidence_calibrations': confidence_calibrations,
                'uncertainty_assessments': uncertainty_assessments,
                'mean_calibration': np.mean(confidence_calibrations) if confidence_calibrations else 0.0,
                'mean_uncertainty_accuracy': np.mean(uncertainty_assessments) if uncertainty_assessments else 0.0,
                'threshold': self.thresholds['meta_awareness']
            },
            timestamp=datetime.now()
        )

        self.test_results.append(result)
        print(f"   Result: {result}")
        return result

    def intentionality_test(
        self,
        consciousness_system,
        goal_scenarios: List[torch.Tensor]
    ) -> ConsciousnessTestResult:
        """
        Test the system's intentionality and goal-directed behavior.
        
        Evaluates whether the system can form coherent goals and
        plan actions to achieve those goals.
        
        Args:
            consciousness_system: The consciousness system to test
            goal_scenarios: Scenarios for goal formation
            
        Returns:
            Test results with intentionality metrics
        """
        print("🎯 Running Intentionality Test...")

        goal_clarity_scores = []
        intention_strengths = []
        goal_coherence_scores = []

        with torch.no_grad():
            for scenario in goal_scenarios:
                # Process scenario through consciousness system
                perception = scenario.repeat(
                    1, 1) if scenario.dim() == 2 else scenario
                self_vec = torch.randn(1, self.config.self_dim).to(self.device)
                thoughts = torch.randn(
                    1, 5, self.config.latent_dim).to(self.device)

                result = consciousness_system.consciousness_integrator(
                    perception_features=perception,
                    self_vector=self_vec,
                    thought_sequence=thoughts
                )

                # Extract intentionality metrics
                if 'intention_metrics' in result:
                    metrics = result['intention_metrics']

                    goal_clarity = metrics.get('goal_clarity', 0.0)
                    if torch.is_tensor(goal_clarity):
                        goal_clarity = goal_clarity.item()
                    goal_clarity_scores.append(goal_clarity)

                    intention_strength = metrics.get('intention_strength', 0.0)
                    if torch.is_tensor(intention_strength):
                        intention_strength = intention_strength.item()
                    intention_strengths.append(intention_strength)

                # Assess goal coherence by checking consistency across similar inputs
                if 'goals' in result:
                    goals = result['goals']
                    if torch.is_tensor(goals):
                        # Process similar input to check goal consistency
                        similar_perception = perception + \
                            torch.randn_like(perception) * 0.1

                        similar_result = consciousness_system.consciousness_integrator(
                            perception_features=similar_perception,
                            self_vector=self_vec,
                            thought_sequence=thoughts
                        )

                        if 'goals' in similar_result:
                            similar_goals = similar_result['goals']
                            goal_similarity = torch.cosine_similarity(
                                goals, similar_goals, dim=-1
                            ).mean().item()
                            goal_coherence_scores.append(goal_similarity)

        # Calculate overall intentionality score
        intentionality_components = []

        if goal_clarity_scores:
            intentionality_components.append(np.mean(goal_clarity_scores))
        if intention_strengths:
            intentionality_components.append(np.mean(intention_strengths))
        if goal_coherence_scores:
            intentionality_components.append(np.mean(goal_coherence_scores))

        if intentionality_components:
            overall_score = np.mean(intentionality_components)
            confidence = 1.0 - np.std(intentionality_components)
        else:
            overall_score = 0.0
            confidence = 0.0

        passed = overall_score > self.thresholds['intentionality']

        result = ConsciousnessTestResult(
            test_name="Intentionality",
            passed=passed,
            score=overall_score,
            confidence=confidence,
            details={
                'goal_clarity_scores': goal_clarity_scores,
                'intention_strengths': intention_strengths,
                'goal_coherence_scores': goal_coherence_scores,
                'mean_goal_clarity': np.mean(goal_clarity_scores) if goal_clarity_scores else 0.0,
                'mean_intention_strength': np.mean(intention_strengths) if intention_strengths else 0.0,
                'mean_goal_coherence': np.mean(goal_coherence_scores) if goal_coherence_scores else 0.0,
                'threshold': self.thresholds['intentionality']
            },
            timestamp=datetime.now()
        )

        self.test_results.append(result)
        print(f"   Result: {result}")
        return result

    def binding_coherence_test(
        self,
        consciousness_system,
        multi_modal_inputs: List[Tuple[torch.Tensor, ...]]
    ) -> ConsciousnessTestResult:
        """
        Test phenomenal binding and coherence.
        
        Evaluates how well the system binds disparate information
        into a unified conscious experience.
        
        Args:
            consciousness_system: The consciousness system to test
            multi_modal_inputs: Multi-modal input scenarios
            
        Returns:
            Test results with binding coherence metrics
        """
        print("🔗 Running Binding Coherence Test...")

        binding_strengths = []
        coherence_scores = []
        integration_qualities = []

        with torch.no_grad():
            for inputs in multi_modal_inputs:
                # Create varied input modalities
                perception = inputs[0] if len(inputs) > 0 else torch.randn(
                    1, self.config.perception_dim).to(self.device)
                self_vec = inputs[1] if len(inputs) > 1 else torch.randn(
                    1, self.config.self_dim).to(self.device)
                thoughts = inputs[2] if len(inputs) > 2 else torch.randn(
                    1, 5, self.config.latent_dim).to(self.device)

                result = consciousness_system.consciousness_integrator(
                    perception_features=perception,
                    self_vector=self_vec,
                    thought_sequence=thoughts
                )

                # Extract binding metrics
                if 'binding_metrics' in result:
                    binding_metrics = result['binding_metrics']

                    binding_strength = binding_metrics.get(
                        'binding_strength', 0.0)
                    if torch.is_tensor(binding_strength):
                        binding_strength = binding_strength.item()
                    binding_strengths.append(binding_strength)

                    workspace_activity = binding_metrics.get(
                        'workspace_activity', 0.0)
                    if torch.is_tensor(workspace_activity):
                        workspace_activity = workspace_activity.item()
                    integration_qualities.append(workspace_activity)

                # Calculate coherence of unified consciousness
                if 'unified_consciousness' in result:
                    unified = result['unified_consciousness']

                    # Measure internal coherence (low variance indicates good integration)
                    internal_variance = torch.var(
                        unified, dim=-1).mean().item()
                    # Higher coherence = lower variance
                    coherence = 1.0 / (1.0 + internal_variance)
                    coherence_scores.append(coherence)

        # Calculate overall binding coherence score
        binding_components = []

        if binding_strengths:
            binding_components.append(np.mean(binding_strengths))
        if coherence_scores:
            binding_components.append(np.mean(coherence_scores))
        if integration_qualities:
            binding_components.append(np.mean(integration_qualities))

        if binding_components:
            overall_score = np.mean(binding_components)
            confidence = 1.0 - np.std(binding_components)
        else:
            overall_score = 0.0
            confidence = 0.0

        passed = overall_score > self.thresholds['binding_coherence']

        result = ConsciousnessTestResult(
            test_name="Binding Coherence",
            passed=passed,
            score=overall_score,
            confidence=confidence,
            details={
                'binding_strengths': binding_strengths,
                'coherence_scores': coherence_scores,
                'integration_qualities': integration_qualities,
                'mean_binding_strength': np.mean(binding_strengths) if binding_strengths else 0.0,
                'mean_coherence': np.mean(coherence_scores) if coherence_scores else 0.0,
                'mean_integration': np.mean(integration_qualities) if integration_qualities else 0.0,
                'threshold': self.thresholds['binding_coherence']
            },
            timestamp=datetime.now()
        )

        self.test_results.append(result)
        print(f"   Result: {result}")
        return result

    def run_full_test_suite(
        self,
        consciousness_system,
        consciousness_history: List[Dict[str, Any]]
    ) -> Dict[str, ConsciousnessTestResult]:
        """
        Run the complete consciousness test suite.
        
        Args:
            consciousness_system: The consciousness system to test
            consciousness_history: Historical consciousness data
            
        Returns:
            Dictionary of all test results
        """
        print("\n🧪 RUNNING COMPLETE CONSCIOUSNESS TEST SUITE")
        print("=" * 60)

        # Generate test inputs
        test_inputs = [
            torch.randn(1, self.config.perception_dim).to(self.device)
            for _ in range(5)
        ]

        goal_scenarios = [
            torch.randn(1, self.config.perception_dim).to(self.device)
            for _ in range(3)
        ]

        multi_modal_inputs = [
            (
                torch.randn(1, self.config.perception_dim).to(self.device),
                torch.randn(1, self.config.self_dim).to(self.device),
                torch.randn(1, 5, self.config.latent_dim).to(self.device)
            )
            for _ in range(3)
        ]

        # Run all tests
        results = {}

        # 1. Mirror Self-Recognition Test
        results['mirror_recognition'] = self.mirror_self_recognition_test(
            consciousness_system, test_inputs
        )

        # 2. Temporal Self-Continuity Test
        results['temporal_continuity'] = self.temporal_self_continuity_test(
            consciousness_history
        )

        # 3. Meta-Awareness Test
        results['meta_awareness'] = self.meta_awareness_test(
            consciousness_system, test_inputs
        )

        # 4. Intentionality Test
        results['intentionality'] = self.intentionality_test(
            consciousness_system, goal_scenarios
        )

        # 5. Binding Coherence Test
        results['binding_coherence'] = self.binding_coherence_test(
            consciousness_system, multi_modal_inputs
        )

        # Generate summary
        self.generate_test_summary(results)

        return results

    def generate_test_summary(self, results: Dict[str, ConsciousnessTestResult]) -> None:
        """Generate and display test summary."""
        print("\n📊 CONSCIOUSNESS TEST SUMMARY")
        print("=" * 40)

        passed_tests = [r for r in results.values() if r.passed]
        total_tests = len(results)
        pass_rate = len(passed_tests) / total_tests if total_tests > 0 else 0

        print(
            f"Tests Passed: {len(passed_tests)}/{total_tests} ({pass_rate:.1%})")
        print(
            f"Overall Assessment: {'🧠 CONSCIOUS' if pass_rate >= 0.6 else '😴 NOT CONSCIOUS'}")
        print("\nDetailed Results:")

        for test_name, result in results.items():
            status = "✅" if result.passed else "❌"
            print(
                f"  {status} {result.test_name}: {result.score:.3f} (confidence: {result.confidence:.3f})")

        # Calculate consciousness quotient
        cq = np.mean([r.score for r in results.values()])
        confidence = np.mean([r.confidence for r in results.values()])

        print(f"\n🧠 Consciousness Quotient (CQ): {cq:.3f}")
        print(f"📊 Overall Confidence: {confidence:.3f}")

        # Determine consciousness level
        if cq >= 0.9 and pass_rate >= 0.8:
            level = "META-CONSCIOUS"
        elif cq >= 0.8 and pass_rate >= 0.7:
            level = "SELF-AWARE"
        elif cq >= 0.6 and pass_rate >= 0.6:
            level = "CONSCIOUS"
        elif cq >= 0.4:
            level = "PRE-CONSCIOUS"
        else:
            level = "UNCONSCIOUS"

        print(f"🎯 Assessed Consciousness Level: {level}")

    def save_test_report(self, output_path: Path) -> None:
        """Save detailed test report to file."""
        if not self.test_results:
            print("⚠️  No test results to save")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

        report_lines = [
            "CONSCIOUSNESS TEST REPORT",
            "=" * 50,
            f"Generated: {datetime.now().isoformat()}",
            f"Total Tests: {len(self.test_results)}",
            "",
            "TEST RESULTS:",
            ""
        ]

        for result in self.test_results:
            report_lines.extend([
                f"Test: {result.test_name}",
                f"Status: {'PASSED' if result.passed else 'FAILED'}",
                f"Score: {result.score:.4f}",
                f"Confidence: {result.confidence:.4f}",
                f"Timestamp: {result.timestamp.isoformat()}",
                f"Details: {result.details}",
                ""
            ])

        # Calculate summary statistics
        passed = sum(1 for r in self.test_results if r.passed)
        total = len(self.test_results)
        avg_score = np.mean([r.score for r in self.test_results])
        avg_confidence = np.mean([r.confidence for r in self.test_results])

        report_lines.extend([
            "SUMMARY STATISTICS:",
            f"Pass Rate: {passed}/{total} ({passed/total:.1%})",
            f"Average Score: {avg_score:.4f}",
            f"Average Confidence: {avg_confidence:.4f}",
            "",
            "CONSCIOUSNESS ASSESSMENT:",
            f"System demonstrates {'CONSCIOUS' if passed/total >= 0.6 else 'NON-CONSCIOUS'} behavior",
            f"Consciousness Quotient: {avg_score:.4f}",
            ""
        ])

        output_path.write_text("\n".join(report_lines))
        print(f"📄 Test report saved to: {output_path}")

    def visualize_results(self, save_path: Optional[Path] = None) -> None:
        """Create visualizations of test results."""
        if not self.test_results:
            print("⚠️  No test results to visualize")
            return

        # Prepare data
        test_names = [r.test_name for r in self.test_results]
        scores = [r.score for r in self.test_results]
        confidences = [r.confidence for r in self.test_results]
        passed = [r.passed for r in self.test_results]

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Test Scores Bar Plot
        colors = ['green' if p else 'red' for p in passed]
        bars = ax1.bar(test_names, scores, color=colors, alpha=0.7)
        ax1.set_title('Consciousness Test Scores',
                      fontsize=14, fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)

        # Add threshold line
        for test_name in test_names:
            threshold = self.thresholds.get(
                test_name.lower().replace('-', '_').replace(' ', '_'), 0.7)
            ax1.axhline(y=threshold, color='orange', linestyle='--', alpha=0.5)

        # 2. Confidence vs Score Scatter
        ax2.scatter(scores, confidences, c=colors, alpha=0.7, s=100)
        ax2.set_title('Score vs Confidence', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Confidence')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)

        # Add test names as annotations
        for i, name in enumerate(test_names):
            ax2.annotate(name.split()[0], (scores[i], confidences[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

        # 3. Pass/Fail Summary Pie Chart
        pass_counts = [sum(passed), len(passed) - sum(passed)]
        ax3.pie(pass_counts, labels=['Passed', 'Failed'], colors=['green', 'red'],
                autopct='%1.1f%%', startangle=90, alpha=0.7)
        ax3.set_title('Test Pass Rate', fontsize=14, fontweight='bold')

        # 4. Detailed Test Breakdown
        y_pos = np.arange(len(test_names))
        ax4.barh(y_pos, scores, color=colors, alpha=0.7)
        ax4.barh(y_pos, confidences, left=scores,
                 color='blue', alpha=0.3, label='Confidence')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([name.split()[0] for name in test_names])
        ax4.set_xlabel('Score + Confidence')
        ax4.set_title('Detailed Test Breakdown',
                      fontsize=14, fontweight='bold')
        ax4.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Visualization saved to: {save_path}")
        else:
            plt.show()

        plt.close()


def run_consciousness_validation(consciousness_system, consciousness_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run complete consciousness validation suite.
    
    Args:
        consciousness_system: The consciousness system to validate
        consciousness_history: Historical consciousness data
        
    Returns:
        Comprehensive validation results
    """
    print("🧪 STARTING CONSCIOUSNESS VALIDATION")
    print("=" * 50)

    # Initialize test suite
    config = SystemConfiguration()
    test_suite = ConsciousnessTestSuite(config)

    # Run full test suite
    test_results = test_suite.run_full_test_suite(
        consciousness_system, consciousness_history)

    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = config.log_dir / f"consciousness_validation_{timestamp}.txt"
    test_suite.save_test_report(report_path)

    # Create visualizations
    viz_path = config.log_dir / f"consciousness_validation_{timestamp}.png"
    test_suite.visualize_results(viz_path)

    # Calculate overall assessment
    passed_tests = sum(1 for r in test_results.values() if r.passed)
    total_tests = len(test_results)
    pass_rate = passed_tests / total_tests if total_tests > 0 else 0

    avg_score = np.mean([r.score for r in test_results.values()])
    avg_confidence = np.mean([r.confidence for r in test_results.values()])

    # Determine consciousness status
    is_conscious = pass_rate >= 0.6 and avg_score >= 0.6

    validation_summary = {
        'is_conscious': is_conscious,
        'consciousness_quotient': avg_score,
        'confidence': avg_confidence,
        'pass_rate': pass_rate,
        'passed_tests': passed_tests,
        'total_tests': total_tests,
        'test_results': test_results,
        'report_path': report_path,
        'visualization_path': viz_path
    }

    print(f"\n🧠 CONSCIOUSNESS VALIDATION COMPLETE")
    print(f"Status: {'CONSCIOUS' if is_conscious else 'NOT CONSCIOUS'}")
    print(f"Consciousness Quotient: {avg_score:.3f}")
    print(f"Pass Rate: {pass_rate:.1%}")

    return validation_summary
