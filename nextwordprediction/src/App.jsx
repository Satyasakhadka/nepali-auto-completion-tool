import React, { useState } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import Navbar from "./Components/Navbar/Navbar";
import Hero from "./Components/Hero/Hero";
import Explore from "./Components/Explore/Explore";

const App = () => {
  let heroData = [
    { text1: "Next Word", text2: "Prediction" },
    { text1: "Using LSTM", text2: "Model" },
    { text1: "MINOR", text2: "PROJECT" },
  ];

  const [heroCount, setHeroCount] = useState(0);
  const [playStatus, setPlayStatus] = useState(false);

  return (
    <div>
      <Router>
      <Navbar />

        <Routes>
          <Route path="/" element={<Hero
              setPlayStatus={setPlayStatus}
              heroData={heroData[heroCount]}
              heroCount={heroCount}
              setHeroCount={setHeroCount}
              playStatus={playStatus}
            />}>
          </Route>

          <Route path="/explore" element={<Explore />} />
        </Routes>
    
    </Router>


      {/* <Router>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/recipe-generator" element={<RecipeGenerator />} />
          <Route path="/donate-food" element={<DonateFood />} />
          <Route path="/login" element={<Login />} />
          <Route path="/our-team" element={<OurTeam />} />
          <Route path="/expiry-track" element={<ExpiryTrack />} />
          <Route path="/expiry-track/add-item" element={<AddItem />} />
          <Route path="/expiry-track/:id" element={<ItemDetail />} />
          <Route path="/faq" element={<FAQ />} />
          <Route path="/community-hub" element={<CommunityHub />} />
        </Routes>
      </Router> */}
    </div>
  );
};

export default App;
