import axios from 'axios';

const apiClient = axios.create({
  baseURL: 'http://127.0.0.1:5000', // バックエンドのURLを指定
  headers: {
    'Content-Type': 'application/json',
  },
});

export default {
  getFeedback() {
    return apiClient.get('/feedback'); // フィードバック生成APIを呼び出し
  },
};
