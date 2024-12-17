import pandas as pd

class FeedbackGenerator:
    def __init__(self, results_file):
        # ユーザー結果を読み込み
        self.results = pd.read_csv(results_file)
    
    def generate_feedback(self):
        feedback = []
        
        # ユーザーごとにフィードバックを生成
        for user_id in self.results['user_id'].unique():
            user_results = self.results[self.results['user_id'] == user_id]
            tags = user_results['tag'].unique()
            
            for tag in tags:
                tag_results = user_results[user_results['tag'] == tag]
                
                correct_rate = tag_results['is_correct'].mean()
                avg_time_per_question = tag_results['time_taken'].mean()
                
                feedback.append(self.generate_suggestion(tag, correct_rate, avg_time_per_question))
        
        return feedback

    def generate_suggestion(self, tag, correct_rate, avg_time):
        if correct_rate < 0.5:
            return f"{tag}の分野の学習が必要です。"
        elif avg_time > 60:
            return f"{tag}に時間がかかっています。効率的な学習方法を検討してください。"
        return f"{tag}は良好な理解ができています。"

# フィードバック生成の実行例
if __name__ == "__main__":
    feedback_generator = FeedbackGenerator('results/results.csv')
    feedback = feedback_generator.generate_feedback()
    
    # フィードバックの表示
    for f in feedback:
        print(f)
