import pandas as pd
import numpy as np
import random

def generate_sample_data():
    """Generate a larger, more diverse dataset of fake and real news for training"""
    
    # Real news headlines - more diverse and realistic
    real_news = [
        # Science and Technology
        "Scientists discover new species of deep-sea creatures in Pacific Ocean",
        "Global climate summit reaches historic agreement on carbon emissions",
        "New vaccine shows promising results in clinical trials",
        "SpaceX successfully launches satellite constellation",
        "World Health Organization releases updated health guidelines",
        "Renewable energy adoption increases by 25% globally",
        "Major tech company announces breakthrough in quantum computing",
        "International peace talks resume in Middle East",
        "NASA plans mission to explore Mars surface",
        "Economic growth exceeds expectations in developing nations",
        "Medical breakthrough: New treatment for rare genetic disorder",
        "Environmental protection laws strengthened in European Union",
        "Breakthrough in fusion energy research announced",
        "Global internet connectivity reaches 70% of world population",
        "New archaeological discovery reveals ancient civilization",
        "Space telescope captures unprecedented images of distant galaxies",
        "International trade agreement benefits multiple countries",
        "Scientific study confirms benefits of Mediterranean diet",
        "Renewable energy costs drop below fossil fuels",
        "World leaders commit to sustainable development goals",
        
        # Politics and Government
        "Parliament passes new education reform bill",
        "President announces infrastructure investment plan",
        "Supreme Court rules on landmark environmental case",
        "International summit addresses global security concerns",
        "New tax policy aims to reduce income inequality",
        "Government launches digital transformation initiative",
        "Parliamentary committee investigates corporate practices",
        "Diplomatic talks progress on trade agreement",
        "Local elections show increased voter participation",
        "Federal budget allocates funds for healthcare improvements",
        
        # Business and Economy
        "Major corporation reports record quarterly profits",
        "Stock market reaches new all-time high",
        "Central bank announces interest rate adjustment",
        "Tech startup receives major investment funding",
        "Automotive industry shifts toward electric vehicles",
        "Retail sales increase during holiday season",
        "Banking sector implements new security measures",
        "Manufacturing sector shows signs of recovery",
        "Real estate market stabilizes after recent fluctuations",
        "Employment rate improves in key economic sectors",
        
        # Health and Medicine
        "Clinical trial shows positive results for cancer treatment",
        "New guidelines issued for diabetes management",
        "Mental health awareness campaign launched nationwide",
        "Hospital implements advanced surgical techniques",
        "Research reveals benefits of regular exercise",
        "Pharmaceutical company develops new antibiotic",
        "Public health officials address vaccination concerns",
        "Medical device approved for widespread use",
        "Study links diet to heart disease prevention",
        "Healthcare workers receive additional training",
        
        # Education and Research
        "University launches new research initiative",
        "Study finds correlation between sleep and academic performance",
        "Educational technology improves student outcomes",
        "Research grant awarded for climate change study",
        "School district implements new curriculum standards",
        "Academic conference addresses global challenges",
        "Student achievement scores show improvement",
        "University partnership promotes international exchange",
        "Research team publishes breakthrough findings",
        "Educational policy reform receives bipartisan support"
    ]
    
    # Fake news headlines - more diverse and realistic sounding
    fake_news = [
        # Outlandish Claims
        "Aliens spotted in downtown New York, government denies everything",
        "Scientists discover that drinking coffee makes you immortal",
        "Secret underground city found beneath Antarctica",
        "Time travel machine invented by high school student",
        "Dragons discovered living in remote mountain caves",
        "Government admits to hiding unicorns in secret facility",
        "New study proves that the Earth is actually flat",
        "Scientists find that chocolate cures all diseases",
        "Secret moon base discovered by amateur astronomer",
        "Time travelers from 3023 warn about impending disaster",
        
        # Conspiracy Theories
        "Ancient pyramids built by advanced alien civilization",
        "Scientists discover that plants can talk to humans",
        "Secret government program creates real superheroes",
        "New technology allows humans to breathe underwater",
        "Ancient scrolls reveal that dinosaurs still exist",
        "Scientists find portal to parallel universe in basement",
        "Government admits to hiding Bigfoot in secret zoo",
        "New study proves that money grows on trees",
        "Secret space program discovers life on Venus",
        "Ancient prophecy predicts end of internet in 2024",
        
        # Impossible Medical Claims
        "Miracle cure discovered: one pill eliminates all diseases",
        "Scientists find that thinking positive thoughts cures cancer",
        "New study shows that breathing air makes you live forever",
        "Medical breakthrough: telepathy now possible with simple device",
        "Doctors discover that eating only pizza prevents all illnesses",
        "Revolutionary treatment: sleep for 24 hours to reverse aging",
        "Scientists prove that watching TV improves eyesight",
        "New vaccine makes people immune to all future diseases",
        "Medical miracle: drinking water cures all mental disorders",
        "Breakthrough: meditation can make you invisible",
        
        # Impossible Technology
        "Inventor creates perpetual motion machine in garage",
        "New smartphone app can read minds",
        "Scientists develop teleportation device using household items",
        "Revolutionary car runs on pure water",
        "Tech company invents device that stops time",
        "New invention allows humans to fly without wings",
        "Scientists create machine that generates infinite energy",
        "Revolutionary app translates animal speech",
        "New technology makes objects disappear instantly",
        "Inventor creates device that predicts lottery numbers",
        
        # Impossible Political Claims
        "Government admits to controlling weather with secret machines",
        "President reveals he is actually a time traveler",
        "Parliament passes law making everyone millionaires",
        "New policy: all citizens get free everything forever",
        "Government discovers way to eliminate all crime instantly",
        "President announces plan to make everyone immortal",
        "New law requires all politicians to be honest",
        "Government invents machine that reads minds",
        "Revolutionary policy: work is now optional for everyone",
        "President reveals secret plan to end all wars forever",
        
        # Impossible Business Claims
        "Company invents money-printing machine for everyone",
        "New business model: customers get paid to buy products",
        "Revolutionary app makes everyone instantly rich",
        "Company discovers way to eliminate all debt worldwide",
        "New technology creates infinite resources",
        "Business breakthrough: free energy for everyone",
        "Company invents device that creates gold from air",
        "Revolutionary service: free everything for life",
        "New app predicts stock market with 100% accuracy",
        "Company discovers way to make money grow on trees"
    ]
    
    # Create DataFrames
    real_df = pd.DataFrame({
        'text': real_news,
        'label': 1  # 1 for real news
    })
    
    fake_df = pd.DataFrame({
        'text': fake_news,
        'label': 0  # 0 for fake news
    })
    
    # Combine and shuffle
    combined_df = pd.concat([real_df, fake_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return combined_df

if __name__ == "__main__":
    # Generate and save sample data
    data = generate_sample_data()
    data.to_csv('sample_news_data.csv', index=False)
    print(f"Generated {len(data)} news articles")
    print(f"Real news: {len(data[data['label'] == 1])}")
    print(f"Fake news: {len(data[data['label'] == 0])}")
    print("Sample data saved to 'sample_news_data.csv'") 