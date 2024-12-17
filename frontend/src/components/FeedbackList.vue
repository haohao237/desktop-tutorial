<template>
    <div class="feedback-list">
      <h2>フィードバック一覧</h2>
      <ul v-if="feedbacks.length">
        <li v-for="(feedback, index) in feedbacks" :key="index">
          <strong>ユーザーID:</strong> {{ feedback.user_id }}<br>
          <strong>タグ:</strong> {{ feedback.tag }}<br>
          <strong>フィードバック:</strong> {{ feedback.feedback }}
        </li>
      </ul>
      <p v-else>フィードバックがまだありません。</p>
    </div>
  </template>
  
  <script>
  import { fetchFeedback } from "../api";
  
  export default {
    name: "FeedbackList",
    data() {
      return {
        feedbacks: [],
      };
    },
    async created() {
      try {
        this.feedbacks = await fetchFeedback();
      } catch (error) {
        console.error("フィードバックの取得中にエラーが発生しました:", error);
      }
    },
  };
  </script>
  
  <style scoped>
  .feedback-list {
    margin: 20px;
    font-family: Arial, sans-serif;
  }
  .feedback-list ul {
    list-style-type: none;
    padding: 0;
  }
  .feedback-list li {
    margin: 10px 0;
    padding: 10px;
    border: 1px solid #ddd;
    border-radius: 5px;
  }
  </style>
  