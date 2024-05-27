import React from 'react';                      
import './Background.css';
import video from '/Users/pc/Desktop/nextwordprediction/src/assets/video.mp4';
import photo1 from '/Users/pc/Desktop/nextwordprediction/src/assets/last.jpg';
import photo2 from '/Users/pc/Desktop/nextwordprediction/src/assets/photo1.jpg';
import photo3 from '/Users/pc/Desktop/nextwordprediction/src/assets/nextw.jpeg';

const Background = ({ playStatus, heroCount }) => {
  return (
    <div className="background-container">
      {playStatus ? (
        <video className='background fade-in' autoPlay loop muted>
          <source src={video} type='video/mp4' />
        </video>
      ) : (
        <img
          src={heroCount === 0 ? photo1 : heroCount === 1 ? photo2 : photo3}
          className='background fade-in'
          alt=""
        />
      )}
    </div>
  );
};

export default Background;
