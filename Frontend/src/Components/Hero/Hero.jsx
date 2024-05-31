import React from 'react';
import './hero.css';
import { Link } from 'react-router-dom';
import arrow_btn from '/Users/pc/Desktop/nextwordprediction/src/assets/arrow_btn.png'
import play_icon from '/Users/pc/Desktop/nextwordprediction/src/assets/play_icon.png'
import pause_icon from '/Users/pc/Desktop/nextwordprediction/src/assets/pause_icon.png'
import Background from '../Background/Background';
const Hero = ({heroData,setHeroCount,heroCount,setPlayStatus,playStatus}) => {
  return (
    <div className='hero'>
      <div className='hero-explore'>
        <p>Next Word predictor</p>
        <img src={arrow_btn} alt='' />
      </div>
      <div className='hero-dot-play'>
        <ul className='hero-dots'>
          <li onClick={()=>setHeroCount(0)} className={heroCount===0?"hero-dot orange":"hero-dot"}></li>
          <li onClick={()=>setHeroCount(1)} className={heroCount===1?"hero-dot orange":"hero-dot"}></li>
          <li onClick={()=>setHeroCount(2)} className={heroCount===2?"hero-dot orange":"hero-dot"}></li>

        </ul>
        <div className="hero-play">
  <img onClick={() =>setPlayStatus(!playStatus)} src={playStatus ? pause_icon : play_icon} alt="/" />
  <p >See the video</p>


  <Background playStatus={playStatus} heroCount={heroCount} />
</div>

        </div>
      </div>
    
    
  );
}

export default Hero;