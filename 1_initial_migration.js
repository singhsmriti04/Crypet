/* eslint-disable no-undef */
const Migrations = artifacts.require("Migrations");
const DaiTokenMock = artifacts.require("DaiTokenMock");

module.exports = async function(deployer) {
  await deployer.deploy(Migrations);
  await deployer.deploy(DaiTokenMock);
  const tokenMock = await DaiTokenMock.deployed()
  
  await tokenMock.mint(
    '0x62c2839Bdabcce85a824f83C2Df5FC4E39497BEB',
    '1000000000000000000000'
  )
};
