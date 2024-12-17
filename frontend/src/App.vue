<template>
    <div id="app">
      <header>
        <h1>AIフィードバックシステム</h1>
      </header>
      <main>
        <FeedbackList />
      </main>
    </div>
  </template>
  
  <script>
  import FeedbackList from "./components/FeedbackList.vue";
  
  export default {
    name: "App",
    components: {
      FeedbackList,
    },
  };
  </script>
  
  <style>
  #app {
    font-family: Arial, sans-serif;
    text-align: center;
    margin: 0 auto;
  }
  header {
    background-color: #282c34;
    color: white;
    padding: 20px;
  }
  main {
    padding: 20px;
  }
  </style>
  