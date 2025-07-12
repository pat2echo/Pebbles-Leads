// CORRECTED FRONTEND JAVASCRIPT - Replace your existing script section


  // ENHANCED STATE MANAGEMENT
  let isLoading = false;
  let messages = [];
  let chatStarted = false;
  let sidebarOpen = false;
  let sessionId = localStorage.getItem('rag-session-id') || generateUUID();
  let currentTopic = 'general';
  let currentQuery = '';
  let currentResponse = '';
  let currentQueryType = '';
  let currentStrategy = '';
  let currentConfidence = 0;
  
  // Store session ID in localStorage
  localStorage.setItem('rag-session-id', sessionId);
  
  // CORRECTED API BASE URL
  const API_BASE_URL = 'http://localhost:8000'; // Changed from 5000 to 8000
  
  // NEW: Utility function to generate UUID
  function generateUUID() {
      return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
          const r = Math.random() * 16 | 0;
          const v = c === 'x' ? r : (r & 0x3 | 0x8);
          return v.toString(16);
      });
  }
  
  // DOM elements (existing + new)
  const chatArea = document.getElementById('chat-area');
  const chatMessagesContainer = document.getElementById('chat-messages-container');
  const chatMessages = document.getElementById('chat-messages');
  const queryForm = document.getElementById('query-form');
  const queryInput = document.getElementById('query-input');
  const sendButton = document.getElementById('send-button');
  const typingIndicator = document.getElementById('typing-indicator');
  const appTitle = document.getElementById('app-title');
  
  // Sidebar elements
  const menuBtn = document.getElementById('menu-btn');
  const sidebar = document.getElementById('sidebar');
  const sidebarOverlay = document.getElementById('sidebar-overlay');
  const closeSidebar = document.getElementById('close-sidebar');
  const indexNav = document.getElementById('index-nav');
  const clearNav = document.getElementById('clear-nav');
  const helpNav = document.getElementById('help-nav');
  
  // Enhanced sidebar elements
  const historyNav = document.getElementById('history-nav');
  const analyticsNav = document.getElementById('analytics-nav');
  const insightsNav = document.getElementById('insights-nav');
  const globalFeedbackNav = document.getElementById('global-feedback-nav');
  const sessionInfo = document.getElementById('session-info');
  const sessionIdDisplay = document.getElementById('session-id-display');
  const currentTopicDisplay = document.getElementById('current-topic-display');
  
  // Modal elements
  const indexModal = document.getElementById('index-modal');
  const indexClose = document.getElementById('index-close');
  const indexForm = document.getElementById('index-form');
  const dataDirInput = document.getElementById('data-dir');
  const indexButton = document.getElementById('index-button');
  const indexStatusDiv = document.getElementById('index-status');
  
  // Enhanced modal elements
  const historyModal = document.getElementById('history-modal');
  const historyClose = document.getElementById('history-close');
  const historyContent = document.getElementById('history-content');
  const analyticsModal = document.getElementById('analytics-modal');
  const analyticsClose = document.getElementById('analytics-close');
  const analyticsContent = document.getElementById('analytics-content');
  const insightsModal = document.getElementById('insights-modal');
  const insightsClose = document.getElementById('insights-close');
  const insightsContent = document.getElementById('insights-content');
  const globalFeedbackModal = document.getElementById('global-feedback-modal');
  const globalFeedbackClose = document.getElementById('global-feedback-close');
  const globalFeedbackContent = document.getElementById('global-feedback-content');
  const detailedFeedbackModal = document.getElementById('detailed-feedback-modal');
  const detailedFeedbackClose = document.getElementById('detailed-feedback-close');
  const topicIndicator = document.getElementById('topic-indicator');
  const topicText = document.getElementById('topic-text');
  
  // Global variables for detailed feedback
  let currentFeedbackTurnId = null;
  let selectedRating = 0;
  
  // Update session info display
  const updateSessionInfo = () => {
      if (sessionIdDisplay) {
          sessionIdDisplay.textContent = `Session: ${sessionId.substring(0, 8)}...`;
      }
      if (currentTopicDisplay) {
          currentTopicDisplay.textContent = `Topic: ${currentTopic.replace('_', ' ')}`;
      }
      if (sessionInfo) {
          sessionInfo.classList.remove('hidden');
      }
  };
  
  // Show topic change notification
  const showTopicChange = (newTopic) => {
      if (topicText) {
          topicText.textContent = `Topic changed to: ${newTopic.replace('_', ' ')}`;
      }
      if (topicIndicator) {
          topicIndicator.classList.remove('hidden');
          setTimeout(() => {
              topicIndicator.classList.add('hidden');
          }, 3000);
      }
  };
  
  // Utility functions
  const getCurrentTime = () => {
      const now = new Date();
      const hours = String(now.getHours()).padStart(2, '0');
      const minutes = String(now.getMinutes()).padStart(2, '0');
      return `${hours}:${minutes}`;
  };
  
  const showTyping = () => {
      typingIndicator.classList.remove('hidden');
      chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
  };
  
  const hideTyping = () => {
      typingIndicator.classList.add('hidden');
  };
  
  const startChat = () => {
      if (!chatStarted) {
          chatStarted = true;
          appTitle.classList.remove('centered');
          appTitle.classList.add('top-positioned');
          chatArea.classList.add('active');
          updateSessionInfo();
      }
  };
  
  // Add message with feedback capability
  const addMessage = (text, isUser = false, turnId = null, confidence = null) => {
      if (!chatStarted) startChat();
      
      const message = { 
          text, 
          isUser, 
          time: getCurrentTime(), 
          turnId,
          confidence,
          feedbackGiven: false
      };
      messages.push(message);
      renderMessages();
  };
  
  // Submit feedback function
  const submitFeedback = async (turnId, score, comments = '') => {
      try {
          const response = await fetch(`${API_BASE_URL}/api/feedback`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                  session_id: sessionId,
                  turn_id: turnId,
                  score: score,
                  comments: comments,
                  query_text: currentQuery,
                  response_text: currentResponse,
                  query_type: currentQueryType,
                  strategy_used: currentStrategy,
                  confidence: currentConfidence
              })
          });
          
          const data = await response.json();
          
          if (data.status === 'success') {
              // Update message to show feedback was given
              const messageIndex = messages.findIndex(m => m.turnId === turnId);
              if (messageIndex !== -1) {
                  messages[messageIndex].feedbackGiven = true;
                  renderMessages();
              }
              
              showNotification('Feedback submitted successfully!', 'success');
          }
      } catch (error) {
          console.error('Error submitting feedback:', error);
          showNotification('Failed to submit feedback', 'error');
      }
  };
  
  // Show notification function
  const showNotification = (message, type = 'info') => {
      const notification = document.createElement('div');
      notification.className = `fixed top-4 right-4 z-50 px-4 py-2 rounded-lg text-white text-sm font-medium transition-all transform translate-x-full ${
          type === 'success' ? 'bg-green-500' : 
          type === 'error' ? 'bg-red-500' : 'bg-blue-500'
      }`;
      notification.textContent = message;
      document.body.appendChild(notification);
      
      // Animate in
      setTimeout(() => {
          notification.classList.remove('translate-x-full');
      }, 100);
      
      // Remove after 3 seconds
      setTimeout(() => {
          notification.classList.add('translate-x-full');
          setTimeout(() => {
              document.body.removeChild(notification);
          }, 300);
      }, 3000);
  };
  
  // Render messages with feedback buttons
  const renderMessages = () => {
      chatMessages.innerHTML = '';
      messages.forEach((message, index) => {
          const messageDiv = document.createElement('div');
          messageDiv.className = `chat-bubble flex ${message.isUser ? 'justify-end' : 'justify-start'} mb-6`;
          
          const confidenceDisplay = message.confidence ? 
              `<div class="text-xs ${message.isUser ? 'text-indigo-200' : 'text-gray-400'} mt-1">
                  Confidence: ${Math.round(message.confidence * 100)}%
              </div>` : '';
          
          const feedbackButtons = !message.isUser && message.turnId && !message.feedbackGiven ? `
              <div class="feedback-buttons mt-3 flex items-center space-x-2">
                  <span class="text-xs text-gray-500">Rate this response:</span>
                  <button onclick="showDetailedFeedbackModal('${message.turnId}')" 
                          class="feedback-btn px-3 py-1 bg-indigo-100 hover:bg-indigo-200 text-indigo-700 rounded-full text-xs font-medium transition-colors">
                      Rate & Comment
                  </button>
                  <button onclick="submitFeedback('${message.turnId}', 1.0, 'Quick positive feedback')" 
                          class="feedback-btn w-6 h-6 rounded-full bg-green-100 hover:bg-green-200 flex items-center justify-center">
                      <svg class="w-3 h-3 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                          <path d="M2 10.5a1.5 1.5 0 113 0v6a1.5 1.5 0 01-3 0v-6zM6 10.333v5.43a2 2 0 001.106 1.79l.05.025A4 4 0 008.943 18h5.416a2 2 0 001.962-1.608l1.2-6A2 2 0 0015.56 8H12V4a2 2 0 00-2-2 1 1 0 00-1 1v.667a4 4 0 01-.8 2.4L6.8 7.933a4 4 0 00-.8 2.4z"/>
                      </svg>
                  </button>
                  <button onclick="submitFeedback('${message.turnId}', 0.2, 'Quick negative feedback')" 
                          class="feedback-btn w-6 h-6 rounded-full bg-red-100 hover:bg-red-200 flex items-center justify-center">
                      <svg class="w-3 h-3 text-red-600" fill="currentColor" viewBox="0 0 20 20">
                          <path d="M18 9.5a1.5 1.5 0 11-3 0v-6a1.5 1.5 0 013 0v6zM14 9.667v-5.43a2 2 0 00-1.106-1.79l-.05-.025A4 4 0 0011.057 2H5.64a2 2 0 00-1.962 1.608l-1.2 6A2 2 0 004.44 12H8v4a2 2 0 002 2 1 1 0 001-1v-.667a4 4 0 01.8-2.4l1.4-1.866a4 4 0 00.8-2.4z"/>
                      </svg>
                  </button>
              </div>
          ` : '';
          
          const feedbackGivenIndicator = message.feedbackGiven ? `
              <div class="text-xs text-green-600 mt-2 flex items-center">
                  <svg class="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                      <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
                  </svg>
                  Feedback submitted
              </div>
          ` : '';
          
          messageDiv.innerHTML = `
              <div class="${message.isUser ? 'message-user' : 'message-assistant'} 
                          rounded-2xl px-5 py-4 max-w-2xl relative shadow-sm">
                  <p class="text-sm leading-relaxed">${message.text}</p>
                  <span class="text-xs ${message.isUser ? 'text-indigo-200' : 'text-gray-500'} 
                        block mt-2 text-right">${message.time}</span>
                  ${confidenceDisplay}
                  ${feedbackButtons}
                  ${feedbackGivenIndicator}
              </div>
          `;
          
          chatMessages.appendChild(messageDiv);
          
          // Show feedback buttons after a delay for assistant messages
          if (!message.isUser && message.turnId && !message.feedbackGiven) {
              setTimeout(() => {
                  const feedbackBtns = messageDiv.querySelector('.feedback-buttons');
                  if (feedbackBtns) {
                      feedbackBtns.classList.add('show');
                  }
              }, 1000);
          }
      });
      
      chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
  };
  
  // Load conversation history
  const loadConversationHistory = async () => {
      try {
          const response = await fetch(`${API_BASE_URL}/api/conversation/${sessionId}?max_turns=20`);
          const data = await response.json();
          
          if (data.turns && data.turns.length > 0) {
              historyContent.innerHTML = `
                  <div class="space-y-4">
                      <div class="bg-blue-50 rounded-lg p-4 mb-6">
                          <h3 class="font-medium text-blue-900 mb-2">Session Overview</h3>
                          <div class="grid grid-cols-2 gap-4 text-sm">
                              <div>
                                  <span class="text-blue-600">Total Messages:</span> ${data.turns.length}
                              </div>
                              <div>
                                  <span class="text-blue-600">Current Topic:</span> ${data.current_topic || 'General'}
                              </div>
                          </div>
                      </div>
                      ${data.turns.map(turn => `
                          <div class="border-l-4 border-indigo-200 pl-4 py-2">
                              <div class="text-sm font-medium text-gray-800 mb-1">
                                  ${turn.query}
                              </div>
                              <div class="text-sm text-gray-600 mb-2">
                                  ${turn.response.substring(0, 200)}${turn.response.length > 200 ? '...' : ''}
                              </div>
                              <div class="flex items-center space-x-4 text-xs text-gray-500">
                                  <span>${new Date(turn.timestamp).toLocaleString()}</span>
                                  <span>Type: ${turn.query_type}</span>
                                  <span>Confidence: ${Math.round(turn.confidence * 100)}%</span>
                                  ${turn.feedback_score ? `<span class="text-green-600">Rated: ${Math.round(turn.feedback_score * 100)}%</span>` : ''}
                              </div>
                          </div>
                      `).join('')}
                  </div>
              `;
          } else {
              historyContent.innerHTML = '<div class="text-center text-gray-500 py-8">No conversation history found.</div>';
          }
      } catch (error) {
          console.error('Error loading history:', error);
          historyContent.innerHTML = '<div class="text-center text-red-500 py-8">Failed to load conversation history.</div>';
      }
  };
  
  // Load analytics
  const loadAnalytics = async () => {
      try {
          const response = await fetch(`${API_BASE_URL}/api/analytics/${sessionId}`);
          const data = await response.json();
          
          if (data.total_feedback > 0) {
              analyticsContent.innerHTML = `
                  <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div class="analytics-card rounded-xl p-4">
                          <h3 class="font-medium text-gray-800 mb-3">Session Stats</h3>
                          <div class="space-y-2 text-sm">
                              <div class="flex justify-between">
                                  <span class="text-gray-600">Total Feedback:</span>
                                  <span class="font-medium">${data.total_feedback}</span>
                              </div>
                              <div class="flex justify-between">
                                  <span class="text-gray-600">Avg Score:</span>
                                  <span class="font-medium">${Math.round(data.average_score * 100)}%</span>
                              </div>
                              <div class="flex justify-between">
                                  <span class="text-gray-600">Avg Confidence:</span>
                                  <span class="font-medium">${Math.round(data.average_confidence * 100)}%</span>
                              </div>
                          </div>
                      </div>
                      
                      <div class="analytics-card rounded-xl p-4">
                          <h3 class="font-medium text-gray-800 mb-3">Query Types</h3>
                          <div class="space-y-2 text-sm">
                              ${Object.entries(data.query_type_distribution || {}).map(([type, count]) => `
                                  <div class="flex justify-between">
                                      <span class="text-gray-600 capitalize">${type}:</span>
                                      <span class="font-medium">${count}</span>
                                  </div>
                              `).join('')}
                          </div>
                      </div>
                  </div>
              `;
          } else {
              analyticsContent.innerHTML = '<div class="text-center text-gray-500 py-8">No analytics data available yet.</div>';
          }
      } catch (error) {
          console.error('Error loading analytics:', error);
          analyticsContent.innerHTML = '<div class="text-center text-red-500 py-8">Failed to load analytics.</div>';
      }
  };
  
  // Load system insights
  const loadSystemInsights = async () => {
      try {
          const [insightsResponse, improvementsResponse] = await Promise.all([
              fetch(`${API_BASE_URL}/api/feedback/insights`),
              fetch(`${API_BASE_URL}/api/feedback/improvements`)
          ]);
          
          const insights = await insightsResponse.json();
          const improvements = await improvementsResponse.json();
          
          if (insights.status === 'success' && improvements.status === 'success') {
              insightsContent.innerHTML = `
                  <div class="space-y-6">
                      <!-- System Improvements Section -->
                      <div class="bg-blue-50 rounded-xl p-6">
                          <h3 class="text-lg font-semibold text-blue-900 mb-4">System Improvement Suggestions</h3>
                          ${improvements.improvements.improvement_suggestions?.length > 0 ? `
                              <div class="space-y-3">
                                  ${improvements.improvements.improvement_suggestions.map(suggestion => `
                                      <div class="flex items-start space-x-3">
                                          <div class="w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                                          <p class="text-blue-800">${suggestion.suggestion}</p>
                                      </div>
                                  `).join('')}
                              </div>
                          ` : '<p class="text-blue-600">No specific improvements identified yet.</p>'}
                      </div>
                      
                      <!-- Strategy Performance Section -->
                      <div class="bg-green-50 rounded-xl p-6">
                          <h3 class="text-lg font-semibold text-green-900 mb-4">Strategy Performance</h3>
                          ${Object.keys(improvements.improvements.strategy_performance || {}).length > 0 ? `
                              <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                                  ${Object.entries(improvements.improvements.strategy_performance).map(([strategy, perf]) => `
                                      <div class="bg-white rounded-lg p-4 border border-green-200">
                                          <h4 class="font-medium text-green-800 capitalize">${strategy}</h4>
                                          <p class="text-sm text-green-600">Avg Score: ${(perf.avg_score * 100).toFixed(1)}%</p>
                                          <p class="text-sm text-green-600">Uses: ${perf.count}</p>
                                      </div>
                                  `).join('')}
                              </div>
                          ` : '<p class="text-green-600">No strategy data available yet.</p>'}
                      </div>
                      
                      <!-- Learning Insights Section -->
                      <div class="bg-purple-50 rounded-xl p-6">
                          <h3 class="text-lg font-semibold text-purple-900 mb-4">Learning Insights</h3>
                          ${insights.insights?.length > 0 ? `
                              <div class="space-y-4">
                                  ${insights.insights.slice(0, 5).map(insight => `
                                      <div class="bg-white rounded-lg p-4 border border-purple-200">
                                          <div class="flex justify-between items-start mb-2">
                                              <h4 class="font-medium text-purple-800 capitalize">
                                                  ${insight.type.replace(/_/g, ' ')}
                                              </h4>
                                              <span class="text-sm bg-purple-100 text-purple-700 px-2 py-1 rounded">
                                                  ${(insight.effectiveness * 100).toFixed(0)}% effective
                                              </span>
                                          </div>
                                          <p class="text-sm text-purple-600">Used ${insight.usage_count} times</p>
                                      </div>
                                  `).join('')}
                              </div>
                          ` : '<p class="text-purple-600">No learning insights available yet.</p>'}
                      </div>
                      
                      <!-- Common Issues Section -->
                      ${improvements.improvements.common_issues?.length > 0 ? `
                          <div class="bg-red-50 rounded-xl p-6">
                              <h3 class="text-lg font-semibold text-red-900 mb-4">Common Issues</h3>
                              <div class="space-y-2">
                                  ${improvements.improvements.common_issues.map(issue => `
                                      <div class="flex justify-between items-center bg-white rounded-lg p-3 border border-red-200">
                                          <p class="text-red-800 text-sm">${issue.issue}</p>
                                          <span class="text-red-600 text-sm font-medium">${issue.frequency}x</span>
                                      </div>
                                  `).join('')}
                              </div>
                          </div>
                      ` : ''}
                  </div>
              `;
          } else {
              insightsContent.innerHTML = '<div class="text-center text-red-500 py-8">Failed to load insights.</div>';
          }
      } catch (error) {
          console.error('Error loading insights:', error);
          insightsContent.innerHTML = '<div class="text-center text-red-500 py-8">Failed to load insights.</div>';
      }
  };
  
  // Load global feedback statistics
  const loadGlobalFeedback = async () => {
      try {
          const response = await fetch(`${API_BASE_URL}/api/feedback/summary/global`);
          const data = await response.json();
          
          if (data.status === 'success') {
              const feedback = data.global_feedback;
              globalFeedbackContent.innerHTML = `
                  <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <!-- Overall Statistics -->
                      <div class="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl p-6">
                          <h3 class="text-lg font-semibold text-indigo-900 mb-4">Overall Statistics</h3>
                          <div class="space-y-3">
                              <div class="flex justify-between">
                                  <span class="text-indigo-700">Total Feedback:</span>
                                  <span class="font-semibold text-indigo-900">${feedback.total_feedback}</span>
                              </div>
                              <div class="flex justify-between">
                                  <span class="text-indigo-700">Average Score:</span>
                                  <span class="font-semibold text-indigo-900">${(feedback.average_score * 100).toFixed(1)}%</span>
                              </div>
                              <div class="flex justify-between">
                                  <span class="text-indigo-700">Learning Insights:</span>
                                  <span class="font-semibold text-indigo-900">${feedback.learning_insights_count || 0}</span>
                              </div>
                          </div>
                      </div>
                      
                      <!-- Score Distribution -->
                      <div class="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl p-6">
                          <h3 class="text-lg font-semibold text-emerald-900 mb-4">Score Distribution</h3>
                          <div class="space-y-3">
                              <div class="flex justify-between items-center">
                                  <span class="text-emerald-700">High (80%+):</span>
                                  <div class="flex items-center space-x-2">
                                      <div class="w-16 bg-emerald-200 rounded-full h-2">
                                          <div class="bg-emerald-500 h-2 rounded-full" style="width: ${feedback.total_feedback > 0 ? (feedback.score_distribution.high / feedback.total_feedback) * 100 : 0}%"></div>
                                      </div>
                                      <span class="font-semibold text-emerald-900">${feedback.score_distribution.high}</span>
                                  </div>
                              </div>
                              <div class="flex justify-between items-center">
                                  <span class="text-emerald-700">Medium (40-80%):</span>
                                  <div class="flex items-center space-x-2">
                                      <div class="w-16 bg-yellow-200 rounded-full h-2">
                                          <div class="bg-yellow-500 h-2 rounded-full" style="width: ${feedback.total_feedback > 0 ? (feedback.score_distribution.medium / feedback.total_feedback) * 100 : 0}%"></div>
                                      </div>
                                      <span class="font-semibold text-emerald-900">${feedback.score_distribution.medium}</span>
                                  </div>
                              </div>
                              <div class="flex justify-between items-center">
                                  <span class="text-emerald-700">Low (<40%):</span>
                                  <div class="flex items-center space-x-2">
                                      <div class="w-16 bg-red-200 rounded-full h-2">
                                          <div class="bg-red-500 h-2 rounded-full" style="width: ${feedback.total_feedback > 0 ? (feedback.score_distribution.low / feedback.total_feedback) * 100 : 0}%"></div>
                                      </div>
                                      <span class="font-semibold text-emerald-900">${feedback.score_distribution.low}</span>
                                  </div>
                              </div>
                          </div>
                      </div>
                  </div>
                  
                  ${Object.keys(feedback.patterns || {}).length > 0 ? `
                      <div class="mt-6 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl p-6">
                          <h3 class="text-lg font-semibold text-purple-900 mb-4">Learning Patterns</h3>
                          <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                              ${Object.entries(feedback.patterns).slice(0, 6).map(([pattern, score]) => `
                                  <div class="bg-white rounded-lg p-3 border border-purple-200">
                                      <p class="text-sm font-medium text-purple-800">${pattern.replace(/_/g, ' ')}</p>
                                      <p class="text-xs text-purple-600">Impact: ${score.toFixed(2)}</p>
                                  </div>
                              `).join('')}
                          </div>
                      </div>
                  ` : ''}
              `;
          } else {
              globalFeedbackContent.innerHTML = '<div class="text-center text-red-500 py-8">Failed to load global feedback.</div>';
          }
      } catch (error) {
          console.error('Error loading global feedback:', error);
          globalFeedbackContent.innerHTML = '<div class="text-center text-red-500 py-8">Failed to load global feedback.</div>';
      }
  };
  
  // Enhanced feedback modal functionality
  const showDetailedFeedbackModal = (turnId) => {
      currentFeedbackTurnId = turnId;
      selectedRating = 0;
      
      const feedbackCommentsEl = document.getElementById('feedback-comments');
      if (feedbackCommentsEl) {
          feedbackCommentsEl.value = '';
      }
      
      // Reset rating buttons
      document.querySelectorAll('.rating-btn').forEach(btn => {
          btn.classList.remove('bg-yellow-500', 'text-white', 'border-yellow-500');
          btn.classList.add('border-gray-300');
      });
      
      if (detailedFeedbackModal) {
          detailedFeedbackModal.classList.remove('hidden');
      }
  };
  
  // Sidebar functionality
  const toggleSidebar = () => {
      sidebarOpen = !sidebarOpen;
      sidebar.classList.toggle('active', sidebarOpen);
      sidebarOverlay.classList.toggle('active', sidebarOpen);
  };
  
  const closeSidebarFn = () => {
      sidebarOpen = false;
      sidebar.classList.remove('active');
      sidebarOverlay.classList.remove('active');
  };
  
  // Event listeners
  if (menuBtn) menuBtn.addEventListener('click', toggleSidebar);
  if (closeSidebar) closeSidebar.addEventListener('click', closeSidebarFn);
  if (sidebarOverlay) sidebarOverlay.addEventListener('click', closeSidebarFn);
  
  if (indexNav) {
      indexNav.addEventListener('click', () => {
          closeSidebarFn();
          if (indexModal) indexModal.classList.remove('hidden');
      });
  }
  
  // History navigation
  if (historyNav) {
      historyNav.addEventListener('click', () => {
          closeSidebarFn();
          if (historyModal) historyModal.classList.remove('hidden');
          loadConversationHistory();
      });
  }
  
  // Analytics navigation
  if (analyticsNav) {
      analyticsNav.addEventListener('click', () => {
          closeSidebarFn();
          if (analyticsModal) analyticsModal.classList.remove('hidden');
          loadAnalytics();
      });
  }
  
  // Insights navigation
  if (insightsNav) {
      insightsNav.addEventListener('click', () => {
          closeSidebarFn();
          if (insightsModal) insightsModal.classList.remove('hidden');
          loadSystemInsights();
      });
  }
  
  // Global feedback navigation
  if (globalFeedbackNav) {
      globalFeedbackNav.addEventListener('click', () => {
          closeSidebarFn();
          if (globalFeedbackModal) globalFeedbackModal.classList.remove('hidden');
          loadGlobalFeedback();
      });
  }
  
  if (clearNav) {
      clearNav.addEventListener('click', () => {
          closeSidebarFn();
          messages = [];
          renderMessages();
          chatStarted = false;
          appTitle.classList.remove('top-positioned');
          appTitle.classList.add('centered');
          chatArea.classList.remove('active');
          // Generate new session ID
          sessionId = generateUUID();
          localStorage.setItem('rag-session-id', sessionId);
          currentTopic = 'general';
          if (sessionInfo) sessionInfo.classList.add('hidden');
      });
  }
  
  if (helpNav) {
      helpNav.addEventListener('click', () => {
          closeSidebarFn();
          addMessage("I'm here to help you with IT support questions. You can ask me about technical issues, software problems, or general IT guidance. I'll remember our conversation and learn from your feedback!", false);
      });
  }
  
  // Modal close handlers
  if (indexClose) {
      indexClose.addEventListener('click', () => {
          if (indexModal) indexModal.classList.add('hidden');
      });
  }
  
  if (historyClose) {
      historyClose.addEventListener('click', () => {
          if (historyModal) historyModal.classList.add('hidden');
      });
  }
  
  if (analyticsClose) {
      analyticsClose.addEventListener('click', () => {
          if (analyticsModal) analyticsModal.classList.add('hidden');
      });
  }
  
  if (insightsClose) {
      insightsClose.addEventListener('click', () => {
          if (insightsModal) insightsModal.classList.add('hidden');
      });
  }
  
  if (globalFeedbackClose) {
      globalFeedbackClose.addEventListener('click', () => {
          if (globalFeedbackModal) globalFeedbackModal.classList.add('hidden');
      });
  }
  
  if (detailedFeedbackClose) {
      detailedFeedbackClose.addEventListener('click', () => {
          if (detailedFeedbackModal) detailedFeedbackModal.classList.add('hidden');
      });
  }
  
  // Rating button functionality
  document.addEventListener('click', (e) => {
      if (e.target.classList.contains('rating-btn')) {
          selectedRating = parseInt(e.target.dataset.rating);
          
          // Update button styles
          document.querySelectorAll('.rating-btn').forEach((btn, index) => {
              if (index < selectedRating) {
                  btn.classList.remove('border-gray-300');
                  btn.classList.add('bg-yellow-500', 'text-white', 'border-yellow-500');
              } else {
                  btn.classList.remove('bg-yellow-500', 'text-white', 'border-yellow-500');
                  btn.classList.add('border-gray-300');
              }
          });
      }
  });
  
  // Cancel detailed feedback
  const cancelDetailedFeedbackBtn = document.getElementById('cancel-detailed-feedback');
  if (cancelDetailedFeedbackBtn) {
      cancelDetailedFeedbackBtn.addEventListener('click', () => {
          if (detailedFeedbackModal) detailedFeedbackModal.classList.add('hidden');
      });
  }
  
  // Enhanced detailed feedback form submission
  const detailedFeedbackForm = document.getElementById('detailed-feedback-form');
  if (detailedFeedbackForm) {
      detailedFeedbackForm.addEventListener('submit', async (e) => {
          e.preventDefault();
          
          if (selectedRating === 0) {
              showNotification('Please select a rating', 'error');
              return;
          }
          
          const feedbackCommentsEl = document.getElementById('feedback-comments');
          const comments = feedbackCommentsEl ? feedbackCommentsEl.value.trim() : '';
          const score = selectedRating / 5.0; // Convert 1-5 to 0.0-1.0
          
          try {
              const response = await fetch(`${API_BASE_URL}/api/feedback`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                      session_id: sessionId,
                      turn_id: currentFeedbackTurnId,
                      score: score,
                      comments: comments,
                      query_text: currentQuery,
                      response_text: currentResponse,
                      query_type: currentQueryType,
                      strategy_used: currentStrategy,
                      confidence: currentConfidence
                  })
              });
              
              const data = await response.json();
              
              if (data.status === 'success') {
                  // Update message to show feedback was given
                  const messageIndex = messages.findIndex(m => m.turnId === currentFeedbackTurnId);
                  if (messageIndex !== -1) {
                      messages[messageIndex].feedbackGiven = true;
                      renderMessages();
                  }
                  
                  if (detailedFeedbackModal) detailedFeedbackModal.classList.add('hidden');
                  showNotification(`Thank you for your ${selectedRating}-star feedback!`, 'success');
              } else {
                  showNotification('Failed to submit feedback', 'error');
              }
          } catch (error) {
              console.error('Error submitting detailed feedback:', error);
              showNotification('Failed to submit feedback', 'error');
          }
      });
  }
  
  // Form submission with session management
  if (queryForm) {
      queryForm.addEventListener('submit', async (e) => {
          e.preventDefault();
          if (isLoading) return;
          
          const query = queryInput.value.trim();
          if (!query) return;
          
          // Store current query for feedback
          currentQuery = query;
          
          addMessage(query, true);
          queryInput.value = '';
          
          isLoading = true;
          showTyping();
          
          try {
              const res = await fetch(`${API_BASE_URL}/api/query`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ 
                      query,
                      session_id: sessionId
                  })
              });
              
              const data = await res.json();
              hideTyping();
              
              if (data.answer) {
                  // Store response data for feedback
                  currentResponse = data.answer;
                  currentQueryType = data.query_type;
                  currentStrategy = data.strategy_used;
                  currentConfidence = data.confidence;
                  
                  // Handle topic changes
                  if (data.topic_changed && data.current_topic !== currentTopic) {
                      currentTopic = data.current_topic;
                      showTopicChange(currentTopic);
                      updateSessionInfo();
                  }
                  
                  // Add message with enhanced data
                  addMessage(
                      data.answer, 
                      false, 
                      data.turn_id, 
                      data.confidence
                  );
              } else {
                  addMessage("Sorry, I couldn't process your request. Please make sure the backend is running.", false);
              }
          } catch (err) {
              hideTyping();
              addMessage("Connection error. Please check if the backend server is running.", false);
          } finally {
              isLoading = false;
          }
      });
  }
  
  // Index form
  if (indexForm) {
      indexForm.addEventListener('submit', async (e) => {
          e.preventDefault();
          if (isLoading) return;
          
          isLoading = true;
          if (indexButton) {
              indexButton.textContent = 'Processing...';
              indexButton.disabled = true;
          }
          if (indexStatusDiv) indexStatusDiv.classList.add('hidden');
          
          try {
              const res = await fetch(`${API_BASE_URL}/api/index`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ data_dir: dataDirInput ? dataDirInput.value : './my_documents' })
              });
              
              const data = await res.json();
              
              if (indexStatusDiv) {
                  indexStatusDiv.innerHTML = `
                      <div class="p-4 rounded-xl glass-card ${data.status === 'success' ? 'text-green-700' : 'text-red-700'}">
                          <p class="font-medium">
                              ${data.status === 'success' 
                                  ? `Successfully indexed ${data.indexed_files?.length || 0} files` 
                                  : `Error: ${data.message}`}
                          </p>
                          ${data.errors && data.errors.length > 0 ? `
                              <ul class="mt-2 text-sm list-disc pl-5">
                                  ${data.errors.map(err => `<li>${err.file}: ${err.error}</li>`).join('')}
                              </ul>
                          ` : ''}
                      </div>
                  `;
                  indexStatusDiv.classList.remove('hidden');
              }
              
          } catch (err) {
              if (indexStatusDiv) {
                  indexStatusDiv.innerHTML = `
                      <div class="p-4 rounded-xl glass-card text-red-700">
                          <p class="font-medium">Failed to index documents. Please check if the backend is running.</p>
                      </div>
                  `;
                  indexStatusDiv.classList.remove('hidden');
              }
          } finally {
              isLoading = false;
              if (indexButton) {
                  indexButton.textContent = 'Index Documents';
                  indexButton.disabled = false;
              }
          }
      });
  }
  
  // Health check on load
  const checkHealth = async () => {
      try {
          const res = await fetch(`${API_BASE_URL}/health`);
          const data = await res.json();
          console.log("Backend connected successfully", data);
      } catch (err) {
          console.log("Backend connection failed", err);
          showNotification('Backend connection failed. Make sure microservices are running on port 8000.', 'error');
      }
  };
  
  // Make functions available globally
  window.submitFeedback = submitFeedback;
  window.showDetailedFeedbackModal = showDetailedFeedbackModal;
  
  // Initialize
  checkHealth();
  updateSessionInfo();
