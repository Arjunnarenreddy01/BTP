import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
from student_query_system import (
    create_student_with_random_history, 
    derive_student_latent_vector, 
    GRADE_MAPPING
)

# Load course factors
course_factors = pd.read_csv("../data/course_latent_factors.csv")

# Feature columns
LATENT_FACTOR_COLS = ['latent_computational_math_heavy', 'latent_theory_heavy', 
                      'latent_difficulty_rigor', 'latent_practicality', 'latent_ease']
RATING_COLS = ['rating_clarity', 'rating_workload', 'rating_interaction', 
               'rating_attendance_strictness', 'rating_assignments', 
               'course_organization', 'overall_rating']
ALL_COURSE_FEATURES = RATING_COLS + LATENT_FACTOR_COLS

class NCFModel(nn.Module):
    """Neural Collaborative Filtering Model for Course Recommendation"""
    
    def __init__(self, student_latent_dim=5, course_feature_dim=12, hidden_dims=[64, 32]):
        super(NCFModel, self).__init__()
        
        self.student_embedding = nn.Linear(student_latent_dim, 64)
        self.course_embedding = nn.Linear(course_feature_dim, 64)
        
        # Concatenated dimension = 64 + 64 = 128
        layers = []
        prev_dim = 128
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.fc = nn.Sequential(*layers)
        
    def forward(self, student_vector, course_features):
        """
        Args:
            student_vector: (batch_size, 5) - student latent vector
            course_features: (batch_size, 12) - course aggregated ratings + latent factors
        
        Returns:
            predictions: (batch_size, 1) - predicted grade score (0-10)
        """
        student_embed = self.student_embedding(student_vector)  # (batch, 64)
        course_embed = self.course_embedding(course_features)   # (batch, 64)
        
        # Concatenate embeddings
        combined = torch.cat([student_embed, course_embed], dim=1)  # (batch, 128)
        
        # Pass through FC layers and return score (0-10)
        score = torch.sigmoid(self.fc(combined)) * 10  # Scale to 0-10
        return score

class NCFRecommender:
    """NCF-based Course Recommender System"""
    
    def __init__(self, device='cpu'):
        self.device = device
        self.model = NCFModel().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
    def train_on_feedback(self, feedback_df, num_epochs=10):
        """
        Train NCF model on feedback data.
        
        Args:
            feedback_df: DataFrame with columns: course_id, overall_rating
            num_epochs: number of training epochs
        """
        # Rename overall_rating in feedback to avoid conflict during merge
        feedback_df = feedback_df.rename(columns={'overall_rating': 'target_rating'})
        
        # Merge feedback with course factors to get features
        course_features_subset = course_factors[['course_id'] + ALL_COURSE_FEATURES].copy()
        train_data = feedback_df.merge(course_features_subset, on='course_id', how='inner')
        
        # For training, we use a simple approach:
        # Sample students from feedback and use their history to train
        # This is a simplified version - in production you'd build proper student profiles
        
        print("🎓 Training NCF Model on Feedback Data...")
        
        for epoch in range(num_epochs):
            total_loss = 0
            batch_count = 0
            
            # Mini-batch training
            for idx in range(0, len(train_data), 32):
                batch = train_data.iloc[idx:idx+32]
                
                # Create dummy student vectors for demonstration
                student_vectors = torch.randn(len(batch), 5).to(self.device)
                course_features = torch.tensor(
                    batch[ALL_COURSE_FEATURES].values, 
                    dtype=torch.float32
                ).to(self.device)
                # Get target ratings
                targets = torch.tensor(
                    batch['target_rating'].values.reshape(-1, 1),
                    dtype=torch.float32
                ).to(self.device)
                
                # Forward pass
                predictions = self.model(student_vectors, course_features)
                loss = self.criterion(predictions, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            avg_loss = total_loss / batch_count
            if (epoch + 1) % 3 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
        
        print("✓ Model trained!\n")
    
    def recommend(self, student_data):
        """
        Recommend courses for a student using NCF.
        
        Args:
            student_data: dict with 'history' (course grades) and 'offered_electives'
        
        Returns:
            DataFrame with course recommendations ranked by predicted score
        """
        # Derive student latent vector
        student_vector = derive_student_latent_vector(student_data, course_factors, LATENT_FACTOR_COLS)
        student_tensor = torch.tensor(student_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Get available courses (OEs)
        oe_courses = student_data['offered_electives']
        course_data = course_factors[course_factors['course_id'].isin(oe_courses)].copy()
        course_data = course_data.sort_values('course_id').reset_index(drop=True)
        
        # Predict scores for all available courses
        course_features = torch.tensor(
            course_data[ALL_COURSE_FEATURES].values,
            dtype=torch.float32
        ).to(self.device)
        
        # Expand student vector to match number of courses
        student_vectors = student_tensor.expand(len(course_data), -1)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(student_vectors, course_features).cpu().numpy()
        
        # Create recommendations dataframe
        recommendations = course_data.copy()
        recommendations['predicted_score'] = predictions.flatten()
        recommendations['recommendation'] = recommendations['predicted_score'].apply(
            lambda x: '✓ Recommended' if x >= 6.5 else '✗ Not Recommended'
        )
        
        # Rank by predicted score
        recommendations = recommendations.sort_values('predicted_score', ascending=False).reset_index(drop=True)
        recommendations['rank'] = range(1, len(recommendations) + 1)
        
        return recommendations

def main():
    """Main function to demonstrate NCF recommendation"""
    
    print("="*80)
    print("NEURAL COLLABORATIVE FILTERING (NCF) COURSE RECOMMENDER")
    print("="*80 + "\n")
    
    # Initialize recommender
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    recommender = NCFRecommender(device=device)
    
    # Load and train on feedback data (simplified)
    feedback_df = pd.read_csv("../data/feedback.csv")
    # Select relevant columns and sample
    feedback_df = feedback_df[['course_id', 'overall_rating']].dropna()
    feedback_df = feedback_df.sample(n=min(1000, len(feedback_df)), random_state=42)
    recommender.train_on_feedback(feedback_df, num_epochs=5)
    
    # Create and recommend for students
    for student_num in range(1, 4):
        student = create_student_with_random_history(num_prev_courses=8, num_oes=40)
        
        print(f"\n{'='*80}")
        print(f"STUDENT #{student_num} RECOMMENDATIONS")
        print(f"{'='*80}\n")
        
        # Display history
        print("📚 COURSE HISTORY:")
        print("-" * 60)
        for course_id in sorted(student['history'].keys()):
            grade = student['history'][course_id]
            score = GRADE_MAPPING[grade]
            print(f"  Course {course_id:3d} → {grade:2s} (Score: {score:.1f})")
        
        # Get recommendations
        recommendations = recommender.recommend(student)
        
        # Display top recommendations
        print("\n\n📋 COURSE RECOMMENDATIONS (Ranked by NCF Model):")
        print("-" * 80)
        print(f"{'Rank':>4} {'Course':>7} {'Predicted Score':>16} {'Status':>18}")
        print("-" * 80)
        
        for idx, row in recommendations.head(15).iterrows():
            rank = row['rank']
            course_id = int(row['course_id'])
            score = row['predicted_score']
            status = row['recommendation']
            print(f"{rank:4d} {course_id:7d} {score:16.2f} {status:>18}")
        
        print("\n... (showing top 15 of 40 courses)")
        
        # Summary
        recommended_count = len(recommendations[recommendations['recommendation'] == '✓ Recommended'])
        not_recommended_count = len(recommendations) - recommended_count
        
        print(f"\n📊 SUMMARY:")
        print(f"  ✓ Recommended: {recommended_count} courses")
        print(f"  ✗ Not Recommended: {not_recommended_count} courses")

if __name__ == "__main__":
    main()
