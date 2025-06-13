# Solving the Traveling Salesman Problem Using Genetic Algorithm and Particle Swarm Optimization

This project explores two nature-inspired optimization techniques — **Genetic Algorithm (GA)** and **Particle Swarm Optimization (PSO)** — to solve the **Traveling Salesman Problem (TSP)**. The TSP is a classic NP-hard problem in computer science where the goal is to determine the shortest possible route that visits a set of cities and returns to the origin city.

## 📄 Project Paper
A full research paper detailing the methodology, experiments, results, and comparative analysis is included in this repository as a PDF file:
- [**Download Paper**](Traveling%20Salesman%20Problem%20Paper.pdf)

## 📂 Files Included
- `gaTSP.py` — Implementation of TSP using Genetic Algorithm
- `psoTSP.py` — Implementation of TSP using Particle Swarm Optimization
- `TSP_Research_Paper.pdf` — Final written report with detailed experiments and results

## 🧪 Experiments Conducted
- Compared GA and PSO using:
  - Varying population/particle sizes
  - Different generation/iteration counts
  - Mutation rate vs best-attractor effect
- Tested performance on:
  - 10-city, 20-city, and 25-city datasets

## 📊 Key Findings
- PSO generally provided shorter route distances and scaled better with larger city sets
- GA required careful parameter tuning (especially mutation rate) to avoid local minima
- Both algorithms were capable of producing near-optimal results but PSO outperformed GA in most configurations

## 👩‍💻 Author
**Erla Hoxha**  
Business Informatics, Epoka University  
📧 erlahoxha04@gmail.com  
📅 February 2025

## 📌 Keywords
`TSP` `Genetic Algorithm` `Particle Swarm Optimization` `NP-hard` `Metaheuristics` `Optimization` `Python`
