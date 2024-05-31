// Explore.js

import React, { useState, useRef } from 'react';
import './Explore.css';

const Explore = () => {
 const [text, setText] = useState('');
 const textareaRef = useRef(null);
 const [predictedWord, setPredictedWord] = useState('');
 const [isLoading, setIsLoading] = useState(false);

 const handleChange = (event) => {
    setText(event.target.value);
 };

 const handlePredictNextWord = async () => {
    try {
      setIsLoading(true); // Set loading state to true while fetching prediction

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ sentence: text }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      console.log('Response from API:', data);

      // Set the predicted word, and let the user decide when to add it
      setPredictedWord(data.predicted_word);
    } catch (error) {
      console.error('Error sending text to API:', error);
    } finally {
      setIsLoading(false); // Set loading state back to false after fetching
    }
 };

 const handleTabPress = (event) => {
    if (event.key === 'Tab') {
      event.preventDefault();
      setText(text + predictedWord);
      setPredictedWord(''); // Clear the predicted word after adding
    }
 };

 return (
    <div className="container">
      <div className="predicted-word">{isLoading ? 'Loading...' : predictedWord}</div>
      <textarea
        ref={textareaRef}
        value={text}
        onChange={handleChange}
        onKeyDown={handleTabPress}
        placeholder="Type Nepali text here..."
      />
      <button onClick={handlePredictNextWord} disabled={isLoading}>
        {isLoading ? 'Loading...' : 'Predict Next Word'}
      </button>
    </div>
 );
};

export default Explore;
