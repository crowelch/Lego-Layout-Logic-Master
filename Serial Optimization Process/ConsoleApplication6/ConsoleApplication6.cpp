// ConsoleApplication6.cpp : Defines the entry point for the console application.
//

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <tchar.h>
#include <cstdio>
#include <ctime>
using namespace std;

static unordered_map<string, unordered_map<string, int>> blocks;
static unordered_map<int, string> colors ;

bool inArray(vector<int> &block, int size)
{

	for (int i = 0; i < block.size(); ++i)
	{
		if (block[i] >= size)
		{
			return false;
		}
	}
	return true;
}

bool equivalentBlockOnebyN(vector<int> &block, int row, int col, vector<vector<int>> &array_Renamed, bool isRowCompare)
{
	if (isRowCompare)
	{
		for (auto b : block)
		{
			if (array_Renamed[row][b] != array_Renamed[row][col])
			{
				return false;
			}
		}
	}
	else
	{
		for (auto b : block)
		{
			if (array_Renamed[b][col] != array_Renamed[row][col])
			{
				return false;
			}
		}
	}
	return true;
}

bool notInQueue(vector<string> &queue, vector<int> &block, int row, int col, bool isRowCompare)
{
	if (isRowCompare)
	{
		for (auto b : block)
		{
			if (find(queue.begin(), queue.end(), to_string(row) + string(",") + to_string(b)) != queue.end())
			{
				return true;
			}
		}
	}
	else
	{
		for (auto b : block)
		{
			if (find(queue.begin(), queue.end(), to_string(b) + string(",") + to_string(col)) != queue.end())
			{
				return true;
			}
		}
	}
	return false;
}

void addToQueue(vector<string> &queue, vector<int> &block, int row, int col, bool isRowCompare)
{
	if (isRowCompare)
	{
		for (auto b : block)
		{
			queue.push_back(to_string(row) + string(",") + to_string(b));
		}
	}
	else
	{
		for (auto b : block)
		{
			queue.push_back(to_string(b) + string(",") + to_string(col));
		}
	}
}

void general(vector<string> &queue, vector<vector<int>> &array_Renamed, int row, int col, int blR, int blC, const string &blockType)
{

	for (int r = 0; r < array_Renamed.size(); r++)
	{
		for (int c = 0; c < array_Renamed[0].size(); c++)
		{

			bool truth = false;
			bool badRow = false;
			bool badCol = false; 
			vector<int> blockC ;
			vector<int> blockR ;
			string temp = to_string(r) + string(",") + to_string(c);
			auto pos = find(queue.begin(), queue.end(), to_string(r) + string(",") + to_string(c));
			if (pos == queue.end() )
			{
				for (int bl = 0; bl < blC; bl++)
				{
					if (bl < col)
					{
						blockC.push_back(c + bl);
					}
					else
					{
						badCol = true;
					}

				}
				for (int bl = 0; bl < blR; bl++)
				{
					if (bl < row)
					{
						blockR.push_back(r + bl);
					}
					else
					{
						badCol = true;
					}
				}

				if (inArray(blockC, col))
				{
					if (equivalentBlockOnebyN(blockC, r, c, array_Renamed, true))
					{
						if (!notInQueue(queue, blockC, r, c, true) & !badRow)
						{
							addToQueue(queue, blockC, r, c, true);
							truth = true;
						}
					}
				}
				if (inArray(blockR, row))
				{
					if (equivalentBlockOnebyN(blockR, r, c, array_Renamed, false))
					{
						if (!notInQueue(queue, blockR, r, c, false))
						{
							addToQueue(queue, blockR, r, c, false & !badCol);
							truth = true;
						}
					}
				}

				if (truth == true)
				{
					try {blocks[blockType][colors[array_Renamed[r][c]]]+= 1;}
					catch(exception e ) { blocks[blockType].insert(make_pair(colors[array_Renamed[r][c]], 1));}
				}
			}
		}

	}
}


void twoByTwo(vector<string> &queue, vector<vector<int>> &array_Renamed, int row, int col)
{

	for (int r = 0; r < array_Renamed.size(); r++)
	{
		for (int c = 0; c < array_Renamed[0].size(); c++)
		{
			bool truth = false;
			if (find(queue.begin(), queue.end(), to_string(r) + string(",") + to_string(c)) == queue.end())
			{
				if (c + 1 < col & r + 1 < row)
				{

					if (array_Renamed[r][c + 1] == array_Renamed[r][c] & array_Renamed[r + 1][c] == array_Renamed[r][c] & array_Renamed[r + 1][c + 1] == array_Renamed[r][c])
					{
						auto pos = find(queue.begin(), queue.end(), to_string(r) + string(",") + to_string(c));
						auto pos1 = find(queue.begin(), queue.end(), to_string(r) + string(",") + to_string(c  + 1));
						auto pos2 = find(queue.begin(), queue.end(), to_string(r + 1) + string(",") + to_string(c ));
						auto pos3 = find(queue.begin(), queue.end(), to_string(r + 1) + string(",") + to_string(c + 1));

						if(pos == queue.end() & pos1 == queue.end() & pos2 == queue.end()  & pos3 == queue.end() ){
							queue.push_back(to_string(r) + string(",") + to_string(c));
							queue.push_back(to_string(r) + string(",") + to_string(c + 1));
							queue.push_back(to_string(r + 1) + string(",") + to_string(c));
							queue.push_back(to_string(r + 1) + string(",") + to_string(c + 1));
							truth = true;
						}
					}
				}

				if (truth == true)
				{

					try {blocks["2x2"][colors[array_Renamed[r][c]]]+= 1;}
					catch(exception e ) { blocks["2x2"].insert(make_pair(colors[array_Renamed[r][c]], 1));}
				}
			}
		}
	}
}

void twoByThree(vector<string> &queue, vector<vector<int>> &array_Renamed, int row, int col)
{

	for (int r = 0; r < array_Renamed.size(); r++)
	{
		for (int c = 0; c < array_Renamed[0].size(); c++)
		{
			bool truth = false;

			if (find(queue.begin(), queue.end(), to_string(r) + string(",") + to_string(c)) == queue.end())
			{
				if (c + 1 < col & c + 2 < col & r + 1 < row)
				{
					if (array_Renamed[r][c + 1] == array_Renamed[r][c] & array_Renamed[r][c + 2] == array_Renamed[r][c] & array_Renamed[r + 1][c] == array_Renamed[r][c] & array_Renamed[r + 1][c + 1] == array_Renamed[r][c] & array_Renamed[r + 1][c + 2] == array_Renamed[r][c])
					{

						auto pos1 = find(queue.begin(), queue.end(), to_string(r) + string(",") + to_string(c  + 1));
						auto pos2 = find(queue.begin(), queue.end(), to_string(r ) + string(",") + to_string(c + 2));
						auto pos3 = find(queue.begin(), queue.end(), to_string(r + 1) + string(",") + to_string(c ));
						auto pos4 = find(queue.begin(), queue.end(), to_string(r + 1) + string(",") + to_string(c + 1));
						auto pos5 = find(queue.begin(), queue.end(), to_string(r + 1) + string(",") + to_string(c + 2));
						if( pos1 == queue.end() & pos2 == queue.end()  & pos3 == queue.end() & pos4 == queue.end() & pos5 == queue.end() ){

							queue.push_back(to_string(r) + string(",") + to_string(c));
							queue.push_back(to_string(r) + string(",") + to_string(c + 1));
							queue.push_back(to_string(r) + string(",") + to_string(c + 2));
							queue.push_back(to_string(r + 1) + string(",") + to_string(c));
							queue.push_back(to_string(r + 1) + string(",") + to_string(c + 1));
							queue.push_back(to_string(r + 1) + string(",") + to_string(c + 2));
							truth = true;
						}
					}
				}
				if (r + 1 < row & r + 2 < row & c + 1 < col)
				{
					if (array_Renamed[r + 1][c] == array_Renamed[r][c] & array_Renamed[r + 2][c] == array_Renamed[r][c] & array_Renamed[r][c + 1] == array_Renamed[r][c] & array_Renamed[r + 1][c + 1] == array_Renamed[r][c] & array_Renamed[r + 2][c + 1] == array_Renamed[r][c])
					{
						auto pos = find(queue.begin(), queue.end(), to_string(r) + string(",") + to_string(c));
						auto pos1 = find(queue.begin(), queue.end(), to_string(r + 1) + string(",") + to_string(c ));
						auto pos2 = find(queue.begin(), queue.end(), to_string(r + 2) + string(",") + to_string(c ));
						auto pos3 = find(queue.begin(), queue.end(), to_string(r) + string(",") + to_string(c + 1));
						auto pos4 = find(queue.begin(), queue.end(), to_string(r + 1) + string(",") + to_string(c + 1));
						auto pos5 = find(queue.begin(), queue.end(), to_string(r + 2) + string(",") + to_string(c + 2));
						if(pos == queue.end() & pos1 == queue.end() & pos2 == queue.end()  & pos3 == queue.end() & pos4 == queue.end() & pos5 == queue.end() ){
							{
								queue.push_back(to_string(r) + string(",") + to_string(c));
								queue.push_back(to_string(r + 1) + string(",") + to_string(c));
								queue.push_back(to_string(r + 2) + string(",") + to_string(c));
								queue.push_back(to_string(r) + string(",") + to_string(c + 1));
								queue.push_back(to_string(r + 1) + string(",") + to_string(c + 1));
								queue.push_back(to_string(r + 2) + string(",") + to_string(c + 1));
								truth = true;
							}
						}
					}

					if (truth == true)
					{
						if (truth == true)
						{
							try {blocks["2x3"][colors[array_Renamed[r][c]]]+= 1;}
							catch(exception e ) { blocks["2x3"].insert(make_pair(colors[array_Renamed[r][c]], 1));}
						}

					}
				}
			}
		}
	}
}

void twoByFour(vector<string> &queue, vector<vector<int>> &array_Renamed, int row, int col)
{
	for (int r = 0; r < array_Renamed.size(); r++)
	{
		for (int c = 0; c < array_Renamed[0].size(); c++)
		{
			bool truth = false;
			if (find(queue.begin(), queue.end(), to_string(r) + string(",") + to_string(c)) == queue.end())
			{
				if (c + 1 < col & c + 2 < col & c + 3 < col & r + 1 < row)
				{
					if (array_Renamed[r][c + 1] == array_Renamed[r][c] & array_Renamed[r][c + 2] == array_Renamed[r][c] & array_Renamed[r][c + 3] == array_Renamed[r][c] & array_Renamed[r + 1][c] == array_Renamed[r][c] & array_Renamed[r + 1][c + 1] == array_Renamed[r][c] & array_Renamed[r + 1][c + 2] == array_Renamed[r][c] & array_Renamed[r + 1][c + 3] == array_Renamed[r][c])
					{
						auto pos = find(queue.begin(), queue.end(), to_string(r) + string(",") + to_string(c));
						auto pos1 = find(queue.begin(), queue.end(), to_string(r) + string(",") + to_string(c  + 1));
						auto pos2 = find(queue.begin(), queue.end(), to_string(r ) + string(",") + to_string(c + 2));
						auto pos3 = find(queue.begin(), queue.end(), to_string(r) + string(",") + to_string(c + 3));
						auto pos4 = find(queue.begin(), queue.end(), to_string(r + 1) + string(",") + to_string(c));
						auto pos5 = find(queue.begin(), queue.end(), to_string(r + 1) + string(",") + to_string(c + 1));
						auto pos6 = find(queue.begin(), queue.end(), to_string(r + 1) + string(",") + to_string(c + 2));
						auto pos7 = find(queue.begin(), queue.end(), to_string(r + 1) + string(",") + to_string(c + 3));


						if(pos == queue.end() & pos1 == queue.end() & pos2 == queue.end()  & pos3 == queue.end() & pos4 == queue.end() & pos5 == queue.end() &  pos6 == queue.end() &  pos7 == queue.end()){

							queue.push_back(to_string(r) + string(",") + to_string(c));
							queue.push_back(to_string(r) + string(",") + to_string(c + 1));
							queue.push_back(to_string(r) + string(",") + to_string(c + 2));
							queue.push_back(to_string(r) + string(",") + to_string(c + 3));
							queue.push_back(to_string(r + 1) + string(",") + to_string(c));
							queue.push_back(to_string(r + 1) + string(",") + to_string(c + 1));
							queue.push_back(to_string(r + 1) + string(",") + to_string(c + 2));
							queue.push_back(to_string(r + 1) + string(",") + to_string(c + 3));
							truth = true;
						}
					}
				}
				if (r + 1  < row & r + 2 < row & r + 3 < row & c + 1 < col)
				{
					if (array_Renamed[r + 1][c] == array_Renamed[r][c] & array_Renamed[r + 2][c] == array_Renamed[r][c] & array_Renamed[r + 3][c] == array_Renamed[r][c] & array_Renamed[r][c + 1] == array_Renamed[r][c] & array_Renamed[r + 1][c + 1] == array_Renamed[r][c] & array_Renamed[r + 2][c + 1] == array_Renamed[r][c] & array_Renamed[r + 3][c + 1] == array_Renamed[r][c])
					{
						auto pos = find(queue.begin(), queue.end(), to_string(r ) + string(",") + to_string(c));
						auto pos1 = find(queue.begin(), queue.end(), to_string(r + 1) + string(",") + to_string(c));
						auto pos2 = find(queue.begin(), queue.end(), to_string(r + 2 ) + string(",") + to_string(c ));
						auto pos3 = find(queue.begin(), queue.end(), to_string(r + 3) + string(",") + to_string(c ));
						auto pos4 = find(queue.begin(), queue.end(), to_string(r ) + string(",") + to_string(c + 1));
						auto pos5 = find(queue.begin(), queue.end(), to_string(r + 1) + string(",") + to_string(c + 1));
						auto pos6 = find(queue.begin(), queue.end(), to_string(r + 2) + string(",") + to_string(c + 1));
						auto pos7 = find(queue.begin(), queue.end(), to_string(r + 3) + string(",") + to_string(c + 1));


						if(pos == queue.end() & pos1 == queue.end() & pos2 == queue.end()  & pos3 == queue.end() & pos4 == queue.end() & pos5 == queue.end() &  pos6 == queue.end() &  pos7 == queue.end()){

							queue.push_back(to_string(r) + string(",") + to_string(c));
							queue.push_back(to_string(r + 1) + string(",") + to_string(c));
							queue.push_back(to_string(r + 2) + string(",") + to_string(c));
							queue.push_back(to_string(r + 3) + string(",") + to_string(c));
							queue.push_back(to_string(r ) + string(",") + to_string(c + 1));
							queue.push_back(to_string(r + 1) + string(",") + to_string(c + 1));
							queue.push_back(to_string(r + 2) + string(",") + to_string(c + 1));
							queue.push_back(to_string(r + 3) + string(",") + to_string(c + 1));
							truth = true;
						}
					}
				}

				if (truth == true)
				{

					try {blocks["2x4"][colors[array_Renamed[r][c]]]+= 1;}
					catch(exception e ) { blocks["2x4"].insert(make_pair(colors[array_Renamed[r][c]], 1));}
				}
			}
		}
	}
}


int _tmain(int argc, _TCHAR* argv[])
{
	std::clock_t start;
    double duration;
    start = std::clock();

	colors.insert(make_pair(1, "black"));
	colors.insert(make_pair(0, "white"));
	colors.insert(make_pair(2, "yellow"));

	blocks.insert(make_pair("2x4", unordered_map<string, int>()));
	blocks.insert(make_pair("2x3", unordered_map<string, int>()));
	blocks.insert(make_pair("2x2", unordered_map<string, int>()));
	blocks.insert(make_pair("1x4", unordered_map<string, int>()));
	blocks.insert(make_pair("1x3", unordered_map<string, int>()));
	blocks.insert(make_pair("1x2", unordered_map<string, int>()));
	blocks.insert(make_pair("1x1", unordered_map<string, int>()));

	int str[13][10] = {     {0,0,0,0,0,0,0,0,0,0},
							{0,0,2,2,2,2,2,0,0,0},
							{0,2,2,2,2,2,2,2,0,0},
							{0,2,1,2,2,1,2,2,2,0},
							{2,2,1,2,2,1,2,2,2,0},
							{2,2,1,2,2,1,2,2,2,0},
							{2,2,2,2,2,2,2,2,2,0},
							{2,2,2,2,2,2,2,2,2,0},
							{0,2,1,1,1,1,1,2,2,0},
							{0,2,1,1,1,1,1,2,2,0},
							{0,2,2,2,2,2,2,2,0,0},
							{0,0,0,2,2,2,2,0,0,0},
							{0,0,0,0,0,0,0,0,0,0}
	};
	vector<vector<int>> array_Renamed(13);
	for (int i = 0; i < 13; ++i){
		array_Renamed[i].resize(10);
	}
	for (int i = 0; i < 13; ++i){
		for (int j = 0; j < 10; ++j){
			array_Renamed[i][j] = str[i][j];
		}
	}

	vector<string> queue =  vector<string>();
	twoByFour(queue, array_Renamed, array_Renamed.size(), array_Renamed[0].size());
	twoByThree(queue, array_Renamed, array_Renamed.size(), array_Renamed[0].size());
	twoByTwo(queue, array_Renamed, array_Renamed.size(), array_Renamed[0].size());

	general(queue, array_Renamed, array_Renamed.size(), array_Renamed[0].size(), 4, 4, "1x4");
	general(queue, array_Renamed, array_Renamed.size(), array_Renamed[0].size(), 3, 3, "1x3");
	general(queue, array_Renamed, array_Renamed.size(), array_Renamed[0].size(), 2, 2, "1x2");
	general(queue, array_Renamed, array_Renamed.size(), array_Renamed[0].size(), 1, 1, "1x1");

	int bl = 0;
	int w = 0;
	for (auto b : blocks)
	{
		cout<< "Block Type  " + b.first << endl;
		for (auto c : b.second)
		{                     

			cout<<"            "+ c.first +  "  : " + to_string(c.second) << endl  ;
			bl += c.second;		
		}
	}

	cout<< "Total Blocks : " + to_string(bl  + w) << endl;
	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;

    std::cout<<"time  "<< duration <<'\n';
	return 0;
}


