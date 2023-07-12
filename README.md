# trust-lending-AI

This project is a variation of a simple trust/money exchange game where one player is now an algorithm. I wrote a machine learning algorithm to act as Player 1 and use previous player data to make predictions about the amount of money that would be returned by Player 2 and send money accordingly.

I incorporated this algorithm into oTree, a platform for experiments in economics, to test my algorithm's ability to accurately make predictions. 

Trust Game:
Player 1 has 10 dollars. They can choose to either send 5 dollars or 10 dollars to Player 2. Player 2 receives triple the amount that Player 1 gave to them (If Player 1 w=sent 10, Player 2 now has 30). Now Player 2 can return a certain amount back to Player 1. The game concludes after this move.
